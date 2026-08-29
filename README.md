# Burmese Voice STT Service

An internal FastAPI microservice that converts uploaded audio into a trustworthy mixed Burmese-English transcript for a banking support copilot. It deliberately contains no RAG, conversation state, answer generation, or TTS.

## Pipeline

`upload validation -> temporary audio preprocessing -> Burmese Whisper ASR -> optional correction provider -> deterministic validation -> final_transcript`

The ASR model is `chuuhtetnaing/whisper-large-v3-myanmar`. Gemini is optional: if it is unavailable, the raw ASR transcript is returned successfully. Protected numeric values are compared deterministically, and corrections that change them or rewrite too much are rejected.

## Microphone preprocessing

Before Whisper, enabled uploads follow: canonical FFmpeg decode (16 kHz mono PCM WAV) → DC offset removal → 80 Hz high-pass filter → optional light FFmpeg `afftdn` noise suppression → capped peak normalization (-3 dBFS, 12 dB maximum gain) → deterministic energy VAD with 250 ms padding → quality gate. All stages are configurable in `.env`; `AUDIO_PREPROCESSING_ENABLED=false` preserves the original upload-to-ASR path for comparisons.

`requirements.audio.txt` contains the three lightweight signal-processing dependencies (`numpy`, `scipy`, and `soundfile`) separately from the ASR requirements, so Docker can keep its existing Torch/Transformers installation layer cached.

The quality gate rejects no speech, severe clipping (more than 2% of samples at/above 0.99 full scale), and extremely low-volume audio. Moderate low volume, clipping, or excessive silence continue with warnings. Preprocessing is not a replacement for a banking terminology prompt or model improvement.

Compare the same real microphone recording across profiles without invoking Whisper:

```bash
python scripts/benchmark_audio_preprocessing.py sample.webm
```

It reports `resample_only`, `resample_vad`, `resample_vad_denoise`, and `full` metrics and timings. To listen to intermediate WAVs during local development, set `AUDIO_DEBUG_SAVE_INTERMEDIATE=true` and optionally change `AUDIO_DEBUG_DIRECTORY`. With Docker, also uncomment the `./debug-audio:/app/debug-audio` bind mount in Compose. These recordings contain customer audio; keep this disabled in production and delete the debug directory after testing.

## Layout

```text
app/
  api/routes/          # health and multipart endpoints
  core/                # settings, exceptions, logging
  providers/           # Gemini and future local correction adapter
  prompts/             # versioned correction prompt
  schemas/             # Pydantic contracts
  services/            # audio, ASR, correction, validation, orchestration
  utils/               # protected-value extraction
tests/
```

## Local setup

Python 3.11+ and FFmpeg are required for default audio normalization. Create `.env` from `.env.example`, set `GEMINI_API_KEY` and `GEMINI_MODEL` to enable correction, then install and run:

```bash
python -m venv .venv
.venv/Scripts/activate  # Windows PowerShell: .venv\\Scripts\\Activate.ps1
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8001
```

Swagger is at `http://localhost:8001/docs`; health is `http://localhost:8001/health`. Docker Compose binds port 8001 to localhost only, so it is reachable from this machine but not directly from the network.

## API

```bash
curl -X POST http://localhost:8001/api/v1/transcribe -F "file=@sample.wav"
```

Success response shape:

```json
{"success":true,"data":{"raw_transcript":"အေတီအမ် ကတ်ရဲ့ ပင် နံပါတ် မေ့သွားတယ်","corrected_transcript":"ATM ကတ်ရဲ့ PIN နံပါတ် မေ့သွားတယ်","final_transcript":"ATM ကတ်ရဲ့ PIN နံပါတ် မေ့သွားတယ်","used_correction":true,"correction_provider":"gemini","validation_passed":true,"warnings":[]}}
```

If correction cannot run, `success` remains true and `final_transcript` equals the raw transcript; `used_correction` is false and `warnings` includes `transcript_correction_unavailable`. If validation detects a changed protected value, it also returns the raw transcript with `protected_value_changed`.

## Docker

```bash
docker build -t burmese-stt-service .
docker run --env-file .env -p 8001:8001 burmese-stt-service
```

The container installs FFmpeg. For CUDA, use a CUDA-compatible base/runtime and set `ASR_DEVICE=cuda`.

Docker Compose mounts a named `huggingface-cache` volume at `/models/huggingface`. The initial ASR request downloads the model there; later container recreations and rebuilds reuse it. Remove that named volume only when you intentionally need to discard cached model files.

## Main backend contract

POST `multipart/form-data` with field `file` to `http://stt-service:8001/api/v1/transcribe`. The main backend must send `data.final_transcript` to its Conversation Manager/RAG pipeline.

## Tests

```bash
pytest
```

Tests mock ASR and correction providers; they never download the Whisper model or call Gemini.

## Limitations and next steps

This synchronous v1 loads Whisper lazily on its first request, and supports non-streaming uploads only. `REQUEST_TIMEOUT_SECONDS` applies an end-to-end request deadline and defaults to 900 seconds to allow the initial Whisper download/load. A timed-out request does not cancel that shared model-loading task. The local correction provider is an explicit extension point, not an implementation. Add model warm-up/readiness checks, terminology registry, metrics/tracing, and internal authentication when the surrounding platform requires them.
