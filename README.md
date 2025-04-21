# Whisper 자동화 및 API 시스템 사용법

본 문서는 두 가지 주요 사용법을 안내합니다.

1.  **자동화 시스템 (권장):** 파일을 업로드하면 시스템이 백그라운드에서 자동으로 처리합니다. 대용량 파일이나 여러 파일을 처리할 때 가장 효율적입니다.
2.  **레거시 API (하위 호환성용):** 파일을 즉시 변환하고 결과를 바로 반환받는 동기식 API입니다. 하위 호환성을 위해 유지되며, 새로운 개발에는 권장하지 않습니다.

## 1. 자동화 시스템 (권장 사용법)

### 1.1. 파일 업로드 및 처리 요청 (`POST /upload`)

음성/영상 파일을 처리 대기열(`input` 폴더)에 추가합니다. 요청이 성공하면 서버는 즉시 `202 Accepted` 상태 코드를 반환하고, 실제 변환 작업은 백그라운드에서 수행됩니다.

```bash
curl -X POST http://localhost:8000/upload \
  -F "file=@/path/to/your/audio.mp3"
```

**응답 예시:**

```json
{
  "message": "파일 수신 완료. 백그라운드에서 처리를 시작합니다.",
  "filename": "audio.mp3"
}
```

### 1.2. 시스템 모델 정보 조회 (`GET /models`)

자동화 시스템에서 사용하는 기본 모델과 허용된 모델 목록을 조회합니다.

```bash
curl -X GET http://localhost:8000/models
```

**응답 예시:**

```json
{
  "default_model": "large-v3",
  "allowed_models": ["tiny", "base", "small", "medium", "large-v3"]
}
```

### 1.3. 결과물 후처리 재실행 (`POST /refresh/{filename}`)

이미 변환이 완료된 `output` 폴더의 `.json` 결과물을 기반으로 `.srt`, `.txt` 파일을 다시 생성할 때 사용합니다.

```bash
curl -X POST http://localhost:8000/refresh/audio.json
```

**응답 예시:**

```json
{
  "message": "'audio.json'에 대한 후처리를 완료했습니다."
}
```

### 1.4. 파일 유효성 검사

시스템은 허용된 확장자를 가진 파일만 처리합니다.

- **허용 확장자:** `.mp3`, `.mp4`, `.mpeg`, `.mpga`, `.m4a`, `.wav`, `.webm`, `.mov`, `.mkv`, `.avi`, `.flac`, `.ogg`
- 허용되지 않는 파일이 `input` 폴더에 들어오면, 자동으로 `rejected` 폴더로 이동되며 처리되지 않습니다.

## 2. 레거시 API (하위 호환성용)

**경고:** 아래의 `/transcribe` 엔드포인트는 곧 지원 중단될 예정입니다. 가급적 `/upload`를 사용한 자동화 시스템을 이용해 주세요.

### 2.1. 파일 즉시 변환 (`POST /transcribe`) - Deprecated

파일을 업로드하면 변환이 완료될 때까지 기다렸다가 결과를 직접 반환받습니다.

#### 기본 요청

```bash
curl -X POST http://localhost:8000/transcribe \
  -F "file=@/path/to/your/audio.mp3"
```

#### 추가 옵션 사용

- `model_name`: 사용할 모델 이름 (기본값: `large-v3`)
- `language`: 오디오 언어 (ISO 639-1 코드, 예: `ko`)
- `task`: 작업 유형 (`transcribe` 또는 `translate`)
- `response_format`: 응답 형식 (`json`, `text`, `srt`, `vtt`)

#### 예제: SRT 형식으로 응답 받기

```bash
curl -X POST http://localhost:8000/transcribe \
  -F "file=@/path/to/your/audio.mp3" \
  -F "response_format=srt"
```

## 3. 공통 정보

### 3.1. API 상태 확인 (`GET /`)

API 서버가 정상 동작하는지 확인합니다.

```bash
curl -X GET http://localhost:8000/
```

**응답 예시:**

```json
{
  "message": "Whisper 자동화 시스템이 정상적으로 동작 중입니다."
}
```

### 3.2. 추가 팁

- **파일 경로:** `@/path/to/your/audio.mp3` 부분은 실제 파일 경로로 수정해야 합니다.
- **응답 저장:** `curl`의 `-o` 옵션을 사용하면 응답 결과를 파일로 저장할 수 있습니다.
  ```bash
  curl -X POST http://localhost:8000/transcribe \
    -F "file=@/path/to/audio.mp3" \
    -F "response_format=srt" \
    -o transcription.srt
  ```
