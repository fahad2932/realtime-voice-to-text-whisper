# vad_utils.py
import torch
import torchaudio

# Load Silero VAD from Torch Hub
# (this will cache the model on first run)
vad_model, utils = torch.hub.load(
    repo_or_dir="snakers4/silero-vad",
    model="silero_vad",
    force_reload=False,
    onnx=False,
)
(get_speech_timestamps, _, read_audio, _, _) = utils


def filter_speech(input_path: str, output_path: str = None) -> torch.Tensor:
    """
    Loads an audio file, runs Silero VAD to identify speech regions,
    stitches them back together, and optionally writes out a new file.
    Returns the filtered waveform tensor (1D, float32).
    """
    # 1) Read & resample to Silero’s expected rate (16 kHz)
    waveform, sr = torchaudio.load(input_path)
    if sr != 16_000:
        waveform = torchaudio.transforms.Resample(sr, 16_000)(waveform)
    waveform = waveform.squeeze(0)  # [time]

    # 2) Get speech timestamps (list of dicts with 'start'/'end' in samples)
    speech_ts = get_speech_timestamps(waveform, vad_model, sampling_rate=16_000)

    # 3) Extract and concat only speech chunks
    speech_chunks = [waveform[s["start"] : s["end"]] for s in speech_ts]
    if not speech_chunks:
        raise RuntimeError("No speech segments detected.")
    filtered = torch.cat(speech_chunks)

    # 4) Optionally write out the filtered file
    if output_path:
        torchaudio.save(output_path, filtered.unsqueeze(0), 16_000)

    return filtered
