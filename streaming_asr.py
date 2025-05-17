# streaming_asr.py

import queue
import threading
import numpy as np
import sounddevice as sd
from faster_whisper import WhisperModel
import torch
from torchaudio.transforms import Resample
from torch.hub import load as torch_hub_load

# 1. Wake-word spotter: tiny Whisper
wake_model = WhisperModel("tiny.en", device="auto", compute_type="int8")
# 2. Main ASR: small model for accuracy
asr_model  = WhisperModel("small.en", device="auto", compute_type="int8")

# 3. Silero VAD for segmenting
vad_model, vad_utils = torch_hub_load(
    "snakers4/silero-vad", "silero_vad", force_reload=False
)
(get_speech_timestamps, _, _, _, _) = vad_utils

# Audio settings
MIC_RATE = 16_000
BLOCK_SEC = 1.0           # process in 1 s blocks
BLOCK_SAMPLES = int(MIC_RATE * BLOCK_SEC)

q = queue.Queue()

def mic_callback(indata, frames, time, status):
    """Puts raw float32 data into a queue, 1 s at a time."""
    if status:
        print(f"Mic status: {status}")
    q.put(indata.copy())

def detect_wake(audio_np: np.ndarray) -> bool:
    """Return True if 'start transcription' appears in the 1 s window."""
    segments, _ = wake_model.transcribe(audio_np,
                                        beam_size=1,
                                        word_timestamps=False,
                                        vad_filter=False)
    text = " ".join(seg.text.lower() for seg in segments)
    return "start transcription" in text

def record_until_silence(streamed_wav: np.ndarray) -> np.ndarray:
    """
    Given some initial audio (the trigger window), keep appending
    blocks until VAD reports >1 s of silence.
    """
    buffer = [streamed_wav]
    silence_acc = 0.0
    while True:
        block = q.get()[:,0]  # mono
        buffer.append(block)
        # run VAD on this 1 s block
        speech_ts = get_speech_timestamps(torch.from_numpy(block),
                                          vad_model,
                                          sampling_rate=MIC_RATE)
        if not speech_ts:
            silence_acc += BLOCK_SEC
        else:
            silence_acc = 0.0
        if silence_acc >= 1.0:
            break
    return np.concatenate(buffer)

def main():
    print("⏳ Waiting for wake-phrase 'start transcription'…")
    # start mic stream
    with sd.InputStream(channels=1,
                        samplerate=MIC_RATE,
                        blocksize=BLOCK_SAMPLES,
                        callback=mic_callback):
        while True:
            block = q.get()[:,0]
            # normalize & to float32
            audio_np = block.astype("float32")

            if detect_wake(audio_np):
                print("✅ Wake-phrase detected — start speaking.")
                speech = record_until_silence(audio_np)
                print("…segment captured, running ASR…")
                segments, info = asr_model.transcribe(
                    speech,
                    vad_filter=False,
                    word_timestamps=True,
                    beam_size=5,
                    batch_size=4,
                )
                # pretty-print
                for seg in segments:
                    print(f"[{seg.start:.2f}-{seg.end:.2f}] {seg.text}")
                print("\n⏳ Back to wake-phrase listening…")

if __name__ == "__main__":
    main()
