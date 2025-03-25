import whisper
import warnings
import torch

warnings.filterwarnings("ignore", message="FP16 is not supported on CPU; using FP32 instead")

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

model = whisper.load_model("large").to(device)

result = model.transcribe("test audio/harvard.wav")

print("Transcription:", result["text"])

translated_result = model.transcribe("test audio/harvard.wav", task="translate")

print("Translation:", translated_result["text"])