import gc
import json
import logging
import os
import shutil
import tempfile
import threading
import time
from contextlib import asynccontextmanager
from functools import lru_cache
from logging.config import dictConfig
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

import torch
import whisper  # 하위호환성을 위해 추가
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse, PlainTextResponse
from faster_whisper import WhisperModel
from pydantic import BaseModel
from pydantic_settings import BaseSettings
from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer

transcription_lock = threading.Lock()
post_processing_locks: Dict[str, threading.Lock] = {}
dict_lock = threading.Lock()


class Settings(BaseSettings):
	ALLOWED_MODELS: List[str] = ['tiny', 'base', 'small', 'medium', 'large-v3']
	DEFAULT_MODEL_NAME: str = 'large-v3'
	DEFAULT_ENGINE: Literal['faster', 'openai'] = 'faster'

	ALLOWED_FILE_EXTENSIONS: List[str] = [
		'.mp3',
		'.mp4',
		'.mpeg',
		'.mpga',
		'.m4a',
		'.wav',
		'.webm',
		'.mov',
		'.mkv',
		'.avi',
		'.flac',
		'.ogg',
	]
	REJECTED_DIR: Path = Path('app/rejected')
	INPUT_DIR: Path = Path('app/input')
	OUTPUT_DIR: Path = Path('app/output')
	PROCESSED_DIR: Path = Path('app/processed')
	UPLOAD_DIR: Path = Path(tempfile.gettempdir()) / 'whisper_uploads'

	def __init__(self, **values):
		super().__init__(**values)
		if self.DEFAULT_MODEL_NAME not in self.ALLOWED_MODELS:
			raise ValueError(
				f'기본 모델({self.DEFAULT_MODEL_NAME})이 허용 목록에 없습니다.'
			)


settings = Settings()


os.makedirs(settings.INPUT_DIR, exist_ok=True)
os.makedirs(settings.OUTPUT_DIR, exist_ok=True)
os.makedirs(settings.PROCESSED_DIR, exist_ok=True)
os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
os.makedirs(settings.REJECTED_DIR, exist_ok=True)


LOGGING_CONFIG = {
	'version': 1,
	'disable_existing_loggers': False,
	'formatters': {
		'default': {
			'()': 'uvicorn.logging.DefaultFormatter',
			'fmt': '%(levelprefix)s %(asctime)s - [%(name)s] - %(message)s',
			'use_colors': True,
			'datefmt': '%Y-%m-%d %H:%M:%S',
		},
		'access': {
			'()': 'uvicorn.logging.AccessFormatter',
			'fmt': '%(levelprefix)s %(asctime)s - %(client_addr)s - "%(request_line)s" %(status_code)s',
			'use_colors': True,
		},
	},
	'handlers': {
		'default': {
			'formatter': 'default',
			'class': 'logging.StreamHandler',
			'stream': 'ext://sys.stderr',
		},
		'access': {
			'formatter': 'access',
			'class': 'logging.StreamHandler',
			'stream': 'ext://sys.stdout',
		},
	},
	'loggers': {
		'': {'handlers': ['default'], 'level': 'INFO', 'propagate': False},
		'uvicorn.error': {'level': 'INFO'},
		'uvicorn.access': {
			'handlers': ['access'],
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
	logger.info(
		f'감지된 GPU: {gpu_info.name} (메모리: {gpu_info.total_memory / 1024**3:.2f}GB)'
	)
	if gpu_info.major >= 8:
		COMPUTE_TYPE = 'int8_float16'
	else:
		COMPUTE_TYPE = 'float16'
else:
	COMPUTE_TYPE = 'int8'
logger.info(f'사용 장치: {DEVICE}, 기본 연산 타입: {COMPUTE_TYPE}')


class AvailableModels(BaseModel):
	default_model: str
	allowed_models: List[str]


class Segment(BaseModel):
	start: float
	end: float
	text: str


class Transcription(BaseModel):
	language: str
	language_probability: Optional[float] = None
	text: str
	segments: List[Segment]


def is_valid_media_file(file_path: Path) -> bool:
	"""파일 확장자가 허용 목록에 있는지 확인합니다."""
	return file_path.suffix.lower() in settings.ALLOWED_FILE_EXTENSIONS


@lru_cache(maxsize=4)
def load_openai_model(name: str) -> whisper.Whisper:
	logger.info(f"OpenAI Whisper 모델 '{name}' 로딩 시작...")
	model = whisper.load_model(name, device=DEVICE)
	logger.info(f"모델 '{name}' 로딩 완료.")
	return model


@lru_cache(maxsize=4)
def load_faster_model(name: str) -> WhisperModel:
	logger.info(
		f"Faster Whisper 모델 '{name}' (연산 타입: {COMPUTE_TYPE}) 로딩 시작..."
	)
	model = WhisperModel(name, device=DEVICE, compute_type=COMPUTE_TYPE)
	logger.info(f"모델 '{name}' 로딩 완료.")
	return model


def format_timestamp(seconds: float, separator: str = '.') -> str:
	total_millis = int(seconds * 1000)
	hours, remainder = divmod(total_millis, 3600000)
	minutes, remainder = divmod(remainder, 60000)
	secs, millis = divmod(remainder, 1000)
	return f'{hours:02d}:{minutes:02d}:{secs:02d}{separator}{millis:03d}'


def save_upload_file_to_temp(upload_file: UploadFile) -> Path:
	suffix = Path(upload_file.filename).suffix
	with tempfile.NamedTemporaryFile(
		delete=False, suffix=suffix, dir=settings.UPLOAD_DIR
	) as tmp:
		shutil.copyfileobj(upload_file.file, tmp)
		return Path(tmp.name)


def create_srt_from_json(json_path: Path):
	srt_path = json_path.with_suffix('.srt')
	try:
		with open(json_path, 'r', encoding='utf-8') as f:
			data = json.load(f)
		lines = [
			f'{i}\n{format_timestamp(seg["start"], ",")} --> {format_timestamp(seg["end"], ",")}\n{seg["text"].strip()}\n\n'
			for i, seg in enumerate(data['segments'], start=1)
		]
		with open(srt_path, 'w', encoding='utf-8') as f:
			f.writelines(lines)
		logger.info(f'SRT 파일 생성 완료: {srt_path}')
	except Exception as e:
		logger.error(f'{srt_path} 생성 중 오류 발생: {e}', exc_info=True)


def create_txt_from_json(json_path: Path):
	txt_path = json_path.with_suffix('.txt')
	try:
		with open(json_path, 'r', encoding='utf-8') as f:
			data = json.load(f)
		full_text = ' '.join(seg['text'].strip() for seg in data['segments'])
		with open(txt_path, 'w', encoding='utf-8') as f:
			f.write(full_text)
		logger.info(f'TXT 파일 생성 완료: {txt_path}')
	except Exception as e:
		logger.error(f'{txt_path} 생성 중 오류 발생: {e}', exc_info=True)


def run_post_processing(original_file_path: Path, result_data: Dict[str, Any]):
	json_output_path = (
		settings.OUTPUT_DIR / original_file_path.name
	).with_suffix('.json')

	with dict_lock:
		if json_output_path.name not in post_processing_locks:
			post_processing_locks[json_output_path.name] = threading.Lock()
		file_lock = post_processing_locks[json_output_path.name]

	with file_lock:
		try:
			with open(json_output_path, 'w', encoding='utf-8') as f:
				json.dump(result_data, f, indent=4, ensure_ascii=False)
			logger.info(f'JSON 결과 저장 완료: {json_output_path}')

			create_srt_from_json(json_output_path)
			create_txt_from_json(json_output_path)

			processed_path = settings.PROCESSED_DIR / original_file_path.name
			shutil.move(str(original_file_path), str(processed_path))
			logger.info(f'원본 파일 이동 완료: {processed_path}')
		except Exception as e:
			logger.error(
				f"'{original_file_path.name}' 후처리 중 오류 발생: {e}",
				exc_info=True,
			)


def run_transcription(file_path: Path):
	if not file_path.exists():
		return

	json_output_path = (settings.OUTPUT_DIR / file_path.name).with_suffix(
		'.json'
	)
	if json_output_path.exists():
		logger.info(
			f'결과가 이미 존재하여 처리를 생략합니다: {json_output_path.name}'
		)
		processed_path = settings.PROCESSED_DIR / file_path.name
		if file_path.exists():
			shutil.move(str(file_path), str(processed_path))
		return

	with transcription_lock:
		logger.info(f"GPU 잠금 획득, '{file_path.name}' 자동 변환 시작.")
		try:
			model = load_faster_model(settings.DEFAULT_MODEL_NAME)

			segments_iterator, info = model.transcribe(
				str(file_path),
				language='ko',
				beam_size=5,
				vad_filter=True,
				temperature=(0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
				word_timestamps=True,
			)

			segments = list(segments_iterator)

			result_data = {
				'language': info.language,
				'language_probability': info.language_probability,
				'duration': info.duration,
				'segments': [],
			}

			for seg in segments:
				result_data['segments'].append(
					{
						'start': seg.start,
						'end': seg.end,
						'text': seg.text,
						'words': [w._asdict() for w in seg.words]
						if seg.words
						else [],
					}
				)

			post_processing_thread = threading.Thread(
				target=run_post_processing,
				args=(file_path, result_data),
				daemon=True,
			)
			post_processing_thread.start()
			logger.info(
				f"'{file_path.name}' 변환 완료, 후처리를 백그라운드에서 시작합니다."
			)

		except Exception as e:
			logger.error(
				f"'{file_path.name}' 변환 중 오류 발생: {e}", exc_info=True
			)
			try:
				rejected_path = settings.REJECTED_DIR / file_path.name
				shutil.move(str(file_path), rejected_path)
				logger.info(
					f"오류가 발생한 파일 '{file_path.name}'을(를) '{rejected_path}'로 이동했습니다."
				)
			except Exception as move_error:
				logger.error(
					f'오류 파일 이동 실패: {move_error}', exc_info=True
				)
		finally:
			logger.info('GPU 잠금 해제, 다음 자동 변환 대기.')


def process_existing_files():
	logger.info(f'기존 파일 스캔 및 처리 시작: {settings.INPUT_DIR}')
	for filename in os.listdir(settings.INPUT_DIR):
		file_path = settings.INPUT_DIR / filename
		if not file_path.is_file():
			continue

		if is_valid_media_file(file_path):
			threading.Thread(
				target=run_transcription, args=(file_path,), daemon=True
			).start()
		else:
			logger.warning(
				f"'{file_path.name}'은(는) 허용되지 않는 파일 형식입니다. "
				f"'{settings.REJECTED_DIR}' 폴더로 이동합니다."
			)
			shutil.move(str(file_path), settings.REJECTED_DIR / file_path.name)


class InputFileHandler(FileSystemEventHandler):
	def on_created(self, event):
		if event.is_directory:
			return

		file_path = Path(event.src_path)
		logger.info(f'새 파일 감지: {file_path}. 복사 완료를 기다립니다...')

		last_size = -1
		retries = 3
		stable_count = 0

		while stable_count < retries:
			try:
				if not file_path.exists():
					logger.warning(
						f"'{file_path}' 확인 중 파일이 사라졌습니다. 처리를 중단합니다."
					)
					return

				current_size = file_path.stat().st_size
				if current_size == last_size and current_size != 0:
					stable_count += 1
				else:
					stable_count = 0

				last_size = current_size
				time.sleep(0.5)

			except FileNotFoundError:
				logger.warning(
					f"'{file_path}' 확인 중 파일을 찾을 수 없습니다. 처리를 중단합니다."
				)
				return
			except Exception as e:
				logger.error(
					f"'{file_path}' 파일 크기 확인 중 오류: {e}", exc_info=True
				)
				return

		logger.info(f"'{file_path.name}' 복사 완료. 처리 스레드를 시작합니다.")

		if is_valid_media_file(file_path):
			threading.Thread(
				target=run_transcription, args=(file_path,), daemon=True
			).start()
		else:
			logger.warning(
				f"'{file_path.name}'은(는) 허용되지 않는 파일 형식입니다. "
				f"'{settings.REJECTED_DIR}' 폴더로 이동합니다."
			)
			shutil.move(str(file_path), settings.REJECTED_DIR / file_path.name)


def start_watching(path: Path):
	event_handler = InputFileHandler()
	observer = Observer()
	observer.schedule(event_handler, str(path), recursive=False)
	observer.start()
	logger.info(f"'{path}' 폴더 감시 시작...")
	try:
		while True:
			time.sleep(5)
	finally:
		observer.stop()
		observer.join()
		logger.info(f"'{path}' 폴더 감시 종료.")


@asynccontextmanager
async def lifespan(app: FastAPI):
	logger.info('애플리케이션 시작...')
	load_faster_model(settings.DEFAULT_MODEL_NAME)
	threading.Thread(target=process_existing_files, daemon=True).start()
	threading.Thread(
		target=start_watching, args=(settings.INPUT_DIR,), daemon=True
	).start()
	yield
	logger.info('애플리케이션 종료...')
	gc.collect()
	if DEVICE == 'cuda':
		torch.cuda.empty_cache()


app = FastAPI(
	title='Whisper 자동화 및 API 시스템',
	description='폴더 감시를 통한 자동 변환과 하위호환성을 위한 동기식 API를 모두 지원합니다.',
	version='0.4.0',  # 버전 업데이트
	lifespan=lifespan,
)


@app.get('/', summary='API 상태 확인')
async def root():
	return {'message': 'Whisper 자동화 시스템이 정상적으로 동작 중입니다.'}


@app.get(
	'/models',
	summary='사용 가능한 모델 목록 조회 (자동화 시스템용)',
	tags=['자동화 시스템'],
)
async def get_available_models():
	return {
		'default_model': settings.DEFAULT_MODEL_NAME,
		'allowed_models': settings.ALLOWED_MODELS,
	}


@app.post(
	'/upload',
	summary='파일을 처리 대기열(input 폴더)에 추가',
	status_code=202,
	tags=['자동화 시스템'],
)
async def upload_file_to_queue(
	file: UploadFile = File(..., description='음성 또는 영상 파일'),
):
	if (
		Path(file.filename).suffix.lower()
		not in settings.ALLOWED_FILE_EXTENSIONS
	):
		raise HTTPException(
			status_code=400,
			detail=f'허용되지 않는 파일 형식입니다. 다음 중 하나여야 합니다: {", ".join(settings.ALLOWED_FILE_EXTENSIONS)}',
		)

	output_path = settings.INPUT_DIR / file.filename
	if output_path.exists():
		return JSONResponse(
			status_code=409,
			content={
				'message': '동일한 이름의 파일이 처리 대기 중입니다.',
				'filename': file.filename,
			},
		)
	try:
		with open(output_path, 'wb') as buffer:
			shutil.copyfileobj(file.file, buffer)
		logger.info(f'API를 통해 파일 수신 및 저장: {output_path}')
		return JSONResponse(
			content={
				'message': '파일 수신 완료. 백그라운드에서 처리를 시작합니다.',
				'filename': file.filename,
			}
		)
	except Exception as e:
		logger.error(f'API 파일 저장 중 오류: {e}', exc_info=True)
		raise HTTPException(
			status_code=500, detail='업로드된 파일을 저장하는데 실패했습니다.'
		)


@app.post(
	'/refresh/{filename}',
	summary='기존 결과물 후처리 재실행',
	tags=['자동화 시스템'],
)
async def refresh_outputs(filename: str):
	json_path = settings.OUTPUT_DIR / Path(filename).with_suffix('.json')
	if not json_path.exists():
		raise HTTPException(
			status_code=404,
			detail=f"'{json_path.name}'에 해당하는 JSON 결과가 존재하지 않습니다.",
		)

	with dict_lock:
		if json_path.name not in post_processing_locks:
			post_processing_locks[json_path.name] = threading.Lock()
		file_lock = post_processing_locks[json_path.name]

	if not file_lock.acquire(blocking=False):
		raise HTTPException(
			status_code=409,
			detail='해당 파일은 현재 다른 작업에서 처리 중입니다.',
		)

	try:
		logger.info(f'후처리 시작: {json_path.name}')
		create_srt_from_json(json_path)
		create_txt_from_json(json_path)
		return {'message': f"'{json_path.name}'에 대한 후처리를 완료했습니다."}
	finally:
		file_lock.release()
		logger.info(f'후처리 완료: {json_path.name}')


@app.post(
	'/transcribe',
	summary='(구) 오디오/비디오 파일 텍스트 변환',
	description='파일을 즉시 변환하고 결과를 반환합니다. 이 엔드포인트는 하위호환성을 위해 유지되며, 향후 제거될 예정입니다. 가급적 `/upload`를 사용한 비동기 방식을 권장합니다.',
	tags=['하위호환성 API'],
	deprecated=True,
)
async def transcribe(
	file: UploadFile = File(..., description='음성 또는 영상 파일'),
	model_name: str = Form('large-v3', description='사용할 Whisper 모델 크기'),
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
	logger.warning("곧 제거될 '/transcribe' 대신 '/upload'를 사용하세요.")
	if not transcription_lock.acquire(blocking=False):
		raise HTTPException(
			status_code=409,
			detail='GPU is currently busy with a background task. Please try again later or use the /upload endpoint.',
		)

	temp_path = None
	try:
		temp_path = save_upload_file_to_temp(file)
		logger.info(
			f"동기 요청 '{file.filename}' 처리 시작 (엔진: {engine}, 포맷: {response_format})"
		)

		transcription_result: Transcription
		if engine == 'faster':
			model = load_faster_model(model_name)
			segments, info = model.transcribe(
				str(temp_path),
				language=language,
				task=task,
				beam_size=5,
				word_timestamps=True,
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
				fp16=(DEVICE == 'cuda'),
			)
			segments = [
				Segment(start=s['start'], end=s['end'], text=s['text'])
				for s in result['segments']
			]
			transcription_result = Transcription(
				language=result['language'],
				text=result['text'],
				segments=segments,
			)

		logger.info(f"동기 요청 '{file.filename}' 처리 완료")

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
		logger.error(f'동기 요청 처리 중 오류 발생: {e}', exc_info=True)
		raise HTTPException(status_code=500, detail=f'내부 서버 오류: {str(e)}')
	finally:
		transcription_lock.release()
		logger.info('동기 요청 처리 완료, GPU 잠금 해제.')

		if temp_path and os.path.exists(temp_path):
			os.remove(temp_path)


if __name__ == '__main__':
	import uvicorn

	uvicorn.run('__main__:app', host='0.0.0.0', port=8000, reload=True)
