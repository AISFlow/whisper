# whisper

## `curl`을 사용한 API 사용법

FastAPI 애플리케이션은 다음과 같은 주요 엔드포인트를 제공합니다:

1. **루트 엔드포인트:** `GET /`
2. **모델 목록 조회:** `GET /models`
3. **파일 전사:** `POST /transcribe`

### 1. 루트 엔드포인트 확인

API가 정상적으로 실행되고 있는지 확인하려면 루트 엔드포인트에 요청을 보냅니다.

```bash
curl -X GET http://localhost:8484/
```

**응답 예시:**
```json
{
  "message": "Welcome to the Enhanced Whisper API!"
}
```

### 2. 사용 가능한 Whisper 모델 목록 조회

Whisper 모델의 목록을 가져오려면 `/models` 엔드포인트에 GET 요청을 보냅니다.

```bash
curl -X GET http://localhost:8484/models
```

**응답 예시:**
```json
{
  "models": ["tiny", "base", "small", "medium", "large", "turbo"]
}
```

### 3. 파일 전사 (Transcribe)

오디오 또는 비디오 파일을 업로드하여 전사 결과를 받으려면 `/transcribe` 엔드포인트에 POST 요청을 보냅니다. 이때 여러 가지 폼 데이터를 함께 전송할 수 있습니다.

#### 기본 전사 요청

```bash
curl -X POST http://localhost:8484/transcribe \
  -F "file=@/path/to/your/audiofile.mp3"
```

#### 추가 옵션 사용

- **모델 선택 (`model_name`):** 사용할 Whisper 모델 이름 (기본값: `base`)
- **작업 유형 (`task`):** 전사 작업 유형 (`transcribe` 또는 `translate`, 기본값: `transcribe`)
- **언어 지정 (`language`):** 오디오의 언어 (예: `en` for English)
- **온도 설정 (`temperature`):** 샘플링 온도 (기본값: `0.0`)
- **응답 형식 (`response_format`):** 응답 형식 (`json`, `text`, `srt`, `vtt`, 기본값: `json`)

##### 예제 1: 모델과 언어를 지정하여 전사

```bash
curl -X POST http://localhost:8484/transcribe \
  -F "file=@/path/to/your/audiofile.mp3" \
  -F "model_name=small" \
  -F "language=en"
```

##### 예제 2: SRT 형식으로 응답 받기

```bash
curl -X POST http://localhost:8484/transcribe \
  -F "file=@/path/to/your/audiofile.mp3" \
  -F "response_format=srt"
```

##### 예제 3: 온도 설정과 번역 작업 수행

```bash
curl -X POST http://localhost:8484/transcribe \
  -F "file=@/path/to/your/audiofile.mp3" \
  -F "task=translate" \
  -F "temperature=0.5"
```

#### 응답 형식별 설명

- **JSON (`json`):** 구조화된 JSON 형식의 응답. 기본 옵션입니다.
  
  **응답 예시:**
  ```json
  {
    "filename": "audiofile.mp3",
    "language": "ko",
    "segments": [
      {
        "start": 0.0,
        "end": 5.0,
        "text": "안녕하세요, 점검하고 있습니다."
      },
      ...
    ]
  }
  ```

- **텍스트 (`text`):** 전사된 텍스트만 반환.

  **응답 예시:**
  ```
  안녕하세요, 점검하고 있습니다.
  ```

- **SRT (`srt`):** 자막 파일 형식으로 반환.

  **응답 예시:**
  ```
  1
  00:00:00,000 --> 00:00:05,000
  안녕하세요, 점검하고 있습니다.

  ...
  ```

- **VTT (`vtt`):** WebVTT 자막 파일 형식으로 반환.

  **응답 예시:**
  ```
  00:00:00.000 --> 00:00:05.000
  안녕하세요, 점검하고 있습니다.

  ...
  ```

### 4. 전체 전사 요청 예제

아래는 모든 옵션을 사용하여 전사를 수행하는 예제입니다.

```bash
curl -X POST http://localhost:8484/transcribe \
  -F "file=@/path/to/your/audiofile.mp3" \
  -F "model_name=medium" \
  -F "task=transcribe" \
  -F "language=en" \
  -F "temperature=0.2" \
  -F "response_format=txt"
```

### 5. 에러 처리

잘못된 요청을 보냈을 경우, API는 적절한 HTTP 상태 코드와 에러 메시지를 반환합니다. 예를 들어, 지원되지 않는 응답 형식을 요청하면 다음과 같은 응답을 받을 수 있습니다.

```bash
curl -X POST http://localhost:8484/transcribe \
  -F "file=@/path/to/your/audiofile.mp3" \
  -F "response_format=unsupported_format"
```

**응답 예시:**
```json
{
  "detail": "Invalid response format."
}
```

## 추가 팁

- **파일 경로:** `@/path/to/your/audiofile.mp3` 부분을 실제 전사할 파일의 경로로 변경하세요.
- **응답 저장:** `curl` 명령어의 출력 결과를 파일로 저장하려면 `-o` 옵션을 사용하세요.

  ```bash
  curl -X POST http://localhost:8484/transcribe \
    -F "file=@/path/to/your/audiofile.mp3" \
    -F "response_format=srt" \
    -o transcription.srt
  ```

- **헤더 설정:** 필요에 따라 추가적인 헤더를 설정할 수 있습니다. 예를 들어, JSON 응답을 명시적으로 요청하려면 `Accept` 헤더를 설정할 수 있습니다.

  ```bash
  curl -X POST http://localhost:8484/transcribe \
    -H "Accept: application/json" \
    -F "file=@/path/to/your/audiofile.mp3"
  ```