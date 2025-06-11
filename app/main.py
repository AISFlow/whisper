# app/main.py

import gc
import logging
import os
import tempfile
from functools import lru_cache
from logging.config import dictConfig
from pathlib import Path
from typing import List, Literal, Optional

import torch
import whisper
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import PlainTextResponse
from faster_whisper import WhisperModel
from pydantic import BaseModel

LOGGING_CONFIG = {
	'version': 1,
	'disable_existing_loggers': False,
	'formatters': {
		'default': {
			'format': '%(asctime)s - [%(levelname)s] - %(name)s - %(message)s',
		},
	},
	'handlers': {
		'default': {
			'formatter': 'default',
			'class': 'logging.StreamHandler',
			'stream': 'ext://sys.stderr',
		},
	},
	'loggers': {
		'__main__': {
			'handlers': ['default'],
			'level': 'INFO',
			'propagate': False,
		},
		'uvicorn': {
			'handlers': ['default'],
			'level': 'INFO',
			'propagate': True,
		},
		'uvicorn.error': {'handlers': ['default'], 'level': 'INFO'},
		'uvicorn.access': {
			'handlers': ['default'],
			'level': 'INFO',
			'propagate': False,
		},
	},
}


dictConfig(LOGGING_CONFIG)

logger = logging.getLogger(__name__)


DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
COMPUTE_TYPE = 'default'
if DEVICE == 'cuda':
	gpu_info = torch.cuda.get_device_properties(0)
	if gpu_info.major >= 8:
		COMPUTE_TYPE = 'int8_float16'
	else:
		COMPUTE_TYPE = 'float16'
else:
	COMPUTE_TYPE = 'int8'

logger.info(f'사용 장치: {DEVICE}, 기본 연산 타입: {COMPUTE_TYPE}')

UPLOAD_DIR = Path(tempfile.gettempdir()) / 'whisper_uploads'
os.makedirs(UPLOAD_DIR, exist_ok=True)


class Segment(BaseModel):
	start: float
	end: float
	text: str


class Transcription(BaseModel):
	language: str
	language_probability: Optional[float] = None
	text: str
	segments: List[Segment]


@lru_cache(maxsize=4)
def load_openai_model(name: str) -> whisper.Whisper:
	return whisper.load_model(name, device=DEVICE)


@lru_cache(maxsize=4)
def load_faster_model(name: str) -> WhisperModel:
	return WhisperModel(name, device=DEVICE, compute_type=COMPUTE_TYPE)


def save_upload_file(upload_file: UploadFile) -> Path:
	suffix = Path(upload_file.filename).suffix
	with tempfile.NamedTemporaryFile(
		delete=False, suffix=suffix, dir=UPLOAD_DIR
	) as tmp:
		tmp.write(upload_file.file.read())
		return Path(tmp.name)


def format_timestamp(seconds: float, separator: str = '.') -> str:
	total_millis = int(seconds * 1000)
	hours, remainder = divmod(total_millis, 3600000)
	minutes, remainder = divmod(remainder, 60000)
	secs, millis = divmod(remainder, 1000)
	return f'{hours:02d}:{minutes:02d}:{secs:02d}{separator}{millis:03d}'


app = FastAPI(
	title='고성능 Whisper API',
	description='다양한 출력 포맷(JSON, Text, VTT, SRT)을 지원하는 Whisper API',
	version='2.2.0',  # 버전 업데이트
)


@app.get('/', summary='API 상태 확인')
async def root():
	return {'message': '고성능 Whisper API가 정상적으로 동작 중입니다.'}


@app.post(
	'/transcribe',
	summary='오디오/비디오 파일 텍스트 변환 (다중 포맷 지원)',
)
async def transcribe(
	file: UploadFile = File(..., description='음성 또는 영상 파일'),
	model_name: str = Form('base', description='사용할 Whisper 모델 크기'),
	language: Optional[str] = Form(
		None, description='오디오 언어 (ISO 639-1 코드)'
	),
	engine: Literal['openai', 'faster'] = Form(
		'faster', description='사용할 추론 엔진'
	),
	task: Literal['transcribe', 'translate'] = Form(
		'transcribe', description='수행할 작업'
	),
	response_format: Literal['json', 'text', 'vtt', 'srt'] = Form(
		'json', description='반환받을 결과 포맷'
	),
):
	temp_path = None
	try:
		temp_path = save_upload_file(file)
		logger.info(
			f"파일 '{file.filename}' 처리 시작 (엔진: {engine}, 포맷: {response_format})"
		)
		transcription_result: Transcription
		if engine == 'faster':
			model = load_faster_model(model_name)
			segments, info = model.transcribe(
				str(temp_path), language=language, task=task, beam_size=5
			)
			segment_list = [
				Segment(start=s.start, end=s.end, text=s.text) for s in segments
			]
			full_text = ' '.join(s.text.strip() for s in segment_list)
			transcription_result = Transcription(
				language=info.language,
				language_probability=info.language_probability,
				text=full_text,
				segments=segment_list,
			)
		else:  # engine == "openai"
			model = load_openai_model(model_name)
			result = model.transcribe(
				str(temp_path),
				language=language,
				task=task,
				fp16=torch.cuda.is_available(),
			)
			transcription_result = Transcription(**result)
		logger.info(f"파일 '{file.filename}' 처리 완료")
		if response_format == 'json':
			return transcription_result.model_dump()
		if response_format == 'text':
			return PlainTextResponse(content=transcription_result.text)
		if response_format == 'vtt':
			separator = '.'
			lines = ['WEBVTT\n']
			for seg in transcription_result.segments:
				start = format_timestamp(seg.start, separator)
				end = format_timestamp(seg.end, separator)
				lines.append(f'{start} --> {end}\n{seg.text.strip()}\n')
			return PlainTextResponse(content=''.join(lines))
		if response_format == 'srt':
			separator = ','
			lines = []
			for i, seg in enumerate(transcription_result.segments, start=1):
				start = format_timestamp(seg.start, separator)
				end = format_timestamp(seg.end, separator)
				lines.append(f'{i}\n{start} --> {end}\n{seg.text.strip()}\n\n')
			return PlainTextResponse(content=''.join(lines))
	except Exception as e:
		logger.exception('처리 중 오류 발생')
		raise HTTPException(status_code=500, detail=f'내부 서버 오류: {str(e)}')
	finally:
		if temp_path and os.path.exists(temp_path):
			os.remove(temp_path)
			gc.collect()
			if DEVICE == 'cuda':
				torch.cuda.empty_cache()


if __name__ == '__main__':
	import uvicorn

	uvicorn.run(
		'main:app',
		host='0.0.0.0',
		port=8000,
		reload=True,
		log_config=LOGGING_CONFIG,
	)
