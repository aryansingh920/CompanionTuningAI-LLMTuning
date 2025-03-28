import whisper
import sounddevice as sd
from scipy.io.wavfile import write

fs = 44100  # Sample rate
seconds = 10  # Duration

print("Recording...")
audio = sd.rec(int(seconds * fs), samplerate=fs, channels=1)
sd.wait()
write("output.wav", fs, audio)
print("Done.")

model = whisper.load_model("base")  # or "small", "medium", "large"
result = model.transcribe("output.wav")
print(result["text"])
