# app/preload/preload_all.py

import gc
import logging
from logging.config import dictConfig
from typing import List

import torch
import typer
import whisper
from faster_whisper import WhisperModel

LOGGING_CONFIG = {
	'version': 1,
	'disable_existing_loggers': False,
	'formatters': {
		'default': {
			'format': '%(asctime)s - [%(levelname)s] - %(name)s - %(message)s'
		}
	},
	'handlers': {
		'default': {
			'formatter': 'default',
			'class': 'logging.StreamHandler',
			'stream': 'ext://sys.stderr',
		}
	},
	'loggers': {
		'__main__': {
			'handlers': ['default'],
			'level': 'INFO',
			'propagate': False,
		},
	},
}
dictConfig(LOGGING_CONFIG)
logger = logging.getLogger(__name__)

FASTER_WHISPER_MODELS = [
	'tiny',
	'base',
	'small',
	'medium',
	'large-v2',
	'large-v3',
]

app = typer.Typer(
	add_completion=False,
	help='Whisper 모델을 미리 다운로드하고 캐싱하여 초기 실행 속도를 높입니다.',
	pretty_exceptions_show_locals=False,
)


def get_device() -> str:
	"""사용 가능한 장치(cuda 또는 cpu)를 반환합니다."""
	return 'cuda' if torch.cuda.is_available() else 'cpu'


@app.command()
def main(
	engine: str = typer.Option(
		'faster',
		'--engine',
		'-e',
		help='사전 로딩할 모델의 엔진을 선택합니다. (faster 또는 openai)',
	),
	model_name: str = typer.Option(
		'all',
		'--model',
		'-m',
		help="특정 모델 이름 또는 'all'을 지정하여 모두 다운로드합니다.",
	),
):
	"""
	지정된 엔진과 모델 이름에 따라 Whisper 모델을 사전 로딩합니다.
	"""
	engine = engine.lower()
	if engine not in ('openai', 'faster'):
		raise typer.BadParameter(
			"engine 값은 'openai' 또는 'faster'여야 합니다."
		)

	device = get_device()
	logger.info(f"선택된 엔진: '{engine}', 장치: '{device}'")

	models_to_process: List[str]
	if model_name.lower() == 'all':
		models_to_process = (
			whisper.available_models()
			if engine == 'openai'
			else FASTER_WHISPER_MODELS
		)
		logger.info(
			f"'{engine}' 엔진의 모든 모델을 순차적으로 처리합니다: {models_to_process}"
		)
	else:
		models_to_process = [model_name]
		logger.info(f"지정된 모델 '{model_name}'을 처리합니다.")

	for name in models_to_process:
		logger.info(f"--- 모델 '{name}' 처리 시작 ---")
		model = None
		try:
			if engine == 'faster':
				model = WhisperModel(name, device=device, compute_type='int8')
			else:  # openai
				model = whisper.load_model(name, device=device)

			logger.info(f"모델 '{name}' 다운로드 및 캐싱 성공.")

		except Exception as e:
			logger.error(f"모델 '{name}' 처리 중 오류 발생: {e}", exc_info=True)

		finally:
			if model:
				del model
				logger.info('메모리에서 모델 객체 삭제 완료.')

			gc.collect()
			logger.info('파이썬 가비지 컬렉션 실행.')
			if device == 'cuda':
				torch.cuda.empty_cache()
				logger.info('PyTorch CUDA 캐시 정리 완료.')
			logger.info(f"--- 모델 '{name}' 처리 완료 ---\n")

	logger.info('모든 요청된 모델의 사전 로딩 작업이 종료되었습니다.')


if __name__ == '__main__':
	app()
