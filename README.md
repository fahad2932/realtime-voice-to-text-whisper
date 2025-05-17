🚀 Real-Time Voice Command ASR Prototype
Real-time automatic speech recognition (ASR) with wake-phrase activation and VAD-driven recording.

🛠️ Features
Continuous Listening: Continuously monitors the default microphone for the wake-phrase "start transcription".

Wake-Phrase Detection: Uses a tiny Whisper model (INT8) via CTranslate2 to spot the wake-phrase within 1 s windows.

VAD-Driven Capture: Once triggered, records in 1 s blocks using Silero VAD. Ends capture after 1 s of silence.

Efficient Transcription: Buffers speech and sends it to a larger Whisper model for high-quality transcription.

Instant Feedback: Prints the transcribed text in real time after each utterance.

