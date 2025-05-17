>Listens continuously on your default microphone
VAD-driven capture: Once triggered, we keep pulling in 1 s blocks and running Silero VAD on them. Any block with zero speech timestamps counts toward “silence.” After 1 s of continuous silence, we consider your utterance done.


>Runs a tiny Whisper model (via CTranslate2) on incoming 1 s windows to spot the wake-phrase “start transcription”
Wake-phrase detection: When it hears that phrase, it switches into VAD-driven recording mode, collecting audio until it sees 1 s of silence. We use a tiny.en model (quantized to INT8) to transcribe each 1 s block as it arrives. If the text contains “start transcription”, we assume you’ve spoken the command.


>Sends that buffered speech chunk to a larger Whisper model for high-quality transcription
Real-time feel: Because we process 1 s windows on the fly, you never wait for the entire file—you speak, and within a second of finishing, you’ll see your text.

>Prints each result and then goes back to listening for the wake-phrase




