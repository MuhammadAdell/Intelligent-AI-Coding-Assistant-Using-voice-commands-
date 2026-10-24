import subprocess
import sys

MODEL_SIZE = "base.en"

WORKER = """
import sys
from faster_whisper import WhisperModel
model = WhisperModel("base.en", compute_type="int8")
# without the VAD pass silence comes back as a hallucinated word ("You"),
# which the assistant then has to answer as if it were a real question
segments, _ = model.transcribe(sys.argv[1], language="en", vad_filter=True)
print(" ".join(segment.text.strip() for segment in segments))
"""


def transcribe(audio_path):
    result = subprocess.run(
        [sys.executable, "-c", WORKER, str(audio_path)],
        capture_output=True, text=True, timeout=300,
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip()[-500:])
    return result.stdout.strip()
