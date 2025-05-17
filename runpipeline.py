from vad_utils import filter_speech
from transcribe import transcribe
from evaluate import compute_wer

# 1) VAD-filter your YouTube clip
filtered = filter_speech("lecture.mp3", output_path="lecture_speech.wav")

# 2) Transcribe
hypothesis, info = transcribe(
    "lecture_speech.wav",
    model_size="small",
    device="cuda",
    vad_filter=False,      # already filtered
    beam_size=5,
    batch_size=8,
)

print(hypothesis)

# 3) Evaluate against your manual transcript
with open("lecture_ref.txt") as f:
    reference = f.read()

print(f"WER: {compute_wer(reference, hypothesis) * 100:.2f}%")
