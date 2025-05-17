# transcribe.py
from faster_whisper import WhisperModel
import numpy as np
import torchaudio

def transcribe(
    audio_input: str,
    model_size: str = "base",
    device: str = "auto",
    vad_filter: bool = False,
    **decode_kwargs
):
    """
    audio_input: path to (raw) audio file or path to VAD-filtered file
    model_size: one of tiny, base, small, medium, large, etc.
    device: "cuda", "cpu" or "auto"
    vad_filter: if True, expect audio_input already speech-filtered, else pass False
    decode_kwargs: forwarded to WhisperModel.transcribe (beam_size, batch_size, etc)
    """
    # 1) Load & optionally filter
    waveform, sr = torchaudio.load(audio_input)
    if sr != 16_000:
        waveform = torchaudio.transforms.Resample(sr, 16_000)(waveform)
    # Convert to numpy for faster-whisper
    audio_np = waveform.squeeze(0).numpy()

    # 2) Instantiate the CTranslate2-backed Whisper
    model = WhisperModel(
        model_size,
        device=device,
        compute_type="int8",       # quantized for speed + memory
    )

    # 3) Transcribe
    segments, info = model.transcribe(
        audio_np,
        vad_filter=vad_filter,
        word_timestamps=True,
        **decode_kwargs
    )

    # 4) Collect plain-text output
    transcript = []
    for segment in segments:
        transcript.append(f"[{segment.start:.2f}-{segment.end:.2f}] {segment.text}")

    return "\n".join(transcript), info
