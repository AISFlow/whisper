import os
import logging
import tempfile
from pathlib import Path
from typing import Optional, List, Union

import torch
import whisper
from fastapi import FastAPI, UploadFile, File, HTTPException, Form, status
from pydantic import BaseModel

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Determine the device to use for model inference
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Directory to store uploaded files temporarily
UPLOAD_DIR = "/tmp"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Default transcription settings
DEFAULT_SETTINGS = {
    "temperature": 0.0,
    "task": "transcribe",
    "verbose": False,
    "language": None,
}

# Define Pydantic models for the API responses
class SegmentResponse(BaseModel):
    start: float
    end: float
    text: str

class TranscriptionResponse(BaseModel):
    filename: str
    language: str
    segments: List[SegmentResponse]

# Initialize the FastAPI application
app = FastAPI()

def load_model(name: str) -> whisper.Whisper:
    """
    Load the specified Whisper model.

    Args:
        name (str): The name of the model to load.

    Returns:
        whisper.Whisper: The loaded Whisper model.

    Raises:
        HTTPException: If the model cannot be loaded.
    """
    try:
        logger.info(f"Loading model: {name}")
        return whisper.load_model(name, device=DEVICE)
    except Exception as e:
        logger.exception("Failed to load model.")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Model '{name}' could not be loaded."
        )

def save_file(file: UploadFile) -> Path:
    """
    Save the uploaded file to a temporary directory.

    Args:
        file (UploadFile): The uploaded file.

    Returns:
        Path: The path to the saved file.

    Raises:
        HTTPException: If the file cannot be saved.
    """
    try:
        suffix = Path(file.filename).suffix
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix, dir=UPLOAD_DIR) as tmp:
            tmp.write(file.file.read())
            logger.info(f"File saved to temporary path: {tmp.name}")
            return Path(tmp.name)
    except Exception as e:
        logger.exception("Failed to save uploaded file.")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="File save failed."
        )

def cleanup(path: Path):
    """
    Remove the temporary file.

    Args:
        path (Path): The path to the file to remove.
    """
    try:
        path.unlink(missing_ok=True)
        logger.info(f"Temporary file {path} removed.")
    except Exception as e:
        logger.warning(f"Cleanup failed for {path}: {e}")

def format_result(result: dict, fmt: str) -> Union[dict, str]:
    """
    Format the transcription result based on the requested format.

    Args:
        result (dict): The transcription result from the model.
        fmt (str): The desired response format.

    Returns:
        Union[dict, str]: The formatted result.

    Raises:
        ValueError: If an unsupported format is requested.
    """
    segments = result.get("segments", [])
    
    if fmt == "txt":
        fmt = "text"

    if fmt == "text":
        return result.get("text", "")
    
    if fmt in {"srt", "vtt"}:
        lines = []
        for i, seg in enumerate(segments, start=1):
            start = format_timestamp(seg["start"], fmt)
            end = format_timestamp(seg["end"], fmt)
            text = seg["text"].strip()
            if fmt == "srt":
                lines.append(f"{i}\n{start} --> {end}\n{text}\n")
            else:  # VTT format
                lines.append(f"{start} --> {end}\n{text}\n")
        return "".join(lines)
    
    # For JSON response, use Pydantic models
    seg_objs = [
        SegmentResponse(start=seg["start"], end=seg["end"], text=seg["text"].strip())
        for seg in segments
    ]
    transcription = TranscriptionResponse(
        filename=result.get("origin_filename", ""),
        language=result.get("language", ""),
        segments=seg_objs
    )
    return transcription.dict()

def format_timestamp(seconds: float, fmt: str) -> str:
    """
    Convert seconds to a timestamp string in SRT or VTT format.

    Args:
        seconds (float): The time in seconds.
        fmt (str): The desired format ('srt' or 'vtt').

    Returns:
        str: The formatted timestamp.
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int(round((seconds - int(seconds)) * 1000))
    
    if fmt == "srt":
        return f"{hours:02}:{minutes:02}:{secs:02},{millis:03}"
    else:  # VTT format uses '.' instead of ','
        return f"{hours:02}:{minutes:02}:{secs:02}.{millis:03}"

@app.get("/models", response_model=dict)
async def get_models():
    """
    Retrieve the list of available Whisper models.

    Returns:
        dict: A dictionary containing the list of models.
    """
    available_models = ["tiny", "base", "small", "medium", "large", "turbo"]
    logger.info("Retrieving available models.")
    return {"models": available_models}

@app.post("/transcribe", response_model=Union[TranscriptionResponse, str])
async def transcribe(
    file: UploadFile = File(...),
    model_name: str = Form("turbo"),
    task: str = Form("transcribe"),
    language: Optional[str] = Form(None),
    temperature: Optional[float] = Form(0.0),
    response_format: Optional[str] = Form("json"),
):
    """
    Transcribe an uploaded audio or video file.

    Args:
        file (UploadFile): The uploaded audio or video file.
        model_name (str): The name of the Whisper model to use.
        task (str): The transcription task ('transcribe' or 'translate').
        language (Optional[str]): The language of the audio.
        temperature (Optional[float]): Sampling temperature.
        response_format (Optional[str]): The format of the response ('json', 'text', 'srt', 'vtt', 'txt').

    Returns:
        Union[TranscriptionResponse, str]: The transcription result in the requested format.

    Raises:
        HTTPException: If an invalid response format is specified or transcription fails.
    """
    allowed_formats = {"json", "text", "srt", "vtt", "txt"}
    if response_format not in allowed_formats:
        logger.error(f"Invalid response format requested: {response_format}")
        raise HTTPException(status_code=400, detail="Invalid response format.")
    
    logger.info(f"Received file: {file.filename} for transcription.")
    temp_path = save_file(file)
    
    try:
        model = load_model(model_name)
        settings = {**DEFAULT_SETTINGS, "task": task, "temperature": temperature, "language": language}
        logger.info(f"Starting transcription for file: {temp_path}")
        result = model.transcribe(str(temp_path), **settings)
        result["origin_filename"] = file.filename
        logger.info("Transcription completed successfully.")
        return format_result(result, response_format)
    except Exception as e:
        logger.exception("Transcription failed.")
        raise HTTPException(status_code=500, detail="Transcription failed.")
    finally:
        cleanup(temp_path)

@app.get("/", response_model=dict)
async def root():
    """
    Root endpoint to verify that the API is running.

    Returns:
        dict: A welcome message.
    """
    return {"message": "Welcome to the Enhanced Whisper API!"}
