import pyaudio
import numpy as np
import wave
import tempfile
import os
import torch
import time
import threading
from faster_whisper import WhisperModel
from transformers import MarianMTModel, MarianTokenizer

FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
CHUNK = 1024
SILENCE_THRESHOLD = 300
MIN_SILENCE_TIME = 0.3
MAX_BUFFER_LENGTH = 5

def remove_repetitions(text):
    if not text or len(text.split()) <= 5:
        return text
    while "  " in text:
        text = text.replace("  ", " ")
    words = text.split()
    cleaned_words = []
    i = 0
    while i < len(words):
        repeat_found = False
        for phrase_len in [5, 4, 3]:
            if i + 2*phrase_len <= len(words):
                phrase1 = " ".join(words[i:i+phrase_len])
                phrase2 = " ".join(words[i+phrase_len:i+2*phrase_len])
                if phrase1 == phrase2:
                    cleaned_words.extend(words[i:i+phrase_len])
                    i += 2*phrase_len
                    repeat_found = True
                    break
        if not repeat_found:
            cleaned_words.append(words[i])
            i += 1
    return " ".join(cleaned_words)

def check_gpu():
    if not torch.cuda.is_available():
        print("❌ CUDA is not available.")
        return False
    print(f"✅ CUDA is available: {torch.cuda.is_available()}")
    print(f"✅ GPU device count: {torch.cuda.device_count()}")
    print(f"✅ Current GPU device: {torch.cuda.current_device()}")
    print(f"✅ GPU device name: {torch.cuda.get_device_name(0)}")
    try:
        x = torch.rand(10, 10).cuda()
        y = x + x
        del x, y
        print("✅ Test CUDA operation successful")
        torch.cuda.empty_cache()
        return True
    except Exception as e:
        print(f"❌ CUDA test failed: {e}")
        return False

class RealTimeTranslator:
    def __init__(self, whisper_model="small", use_gpu=True):
        gpu_available = check_gpu()
        if gpu_available and use_gpu:
            os.environ["CUDA_VISIBLE_DEVICES"] = "0"
            self.device = "cuda:0"
            self.whisper_device = "cuda"
            torch.cuda.set_device(0)
            print(f"GPU detected: {torch.cuda.get_device_name(0)}")
            print(f"GPU memory before model loading: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")
            torch.cuda.empty_cache()
        else:
            print("⚠️ Running on CPU")
            self.device = "cpu"
            self.whisper_device = "cpu"
        print(f"Using device: {self.device}")
        self.audio = pyaudio.PyAudio()
        self.stream = None
        print("Loading speech recognition model...")
        compute_type = "float16" if self.device.startswith("cuda") else "int8"
        self.asr_model = WhisperModel(
            whisper_model, 
            device=self.whisper_device,
            compute_type=compute_type,
            download_root="./models"
        )
        print("Loading translation model...")
        self.trans_model_name = "Helsinki-NLP/opus-mt-mul-en"
        self.trans_tokenizer = MarianTokenizer.from_pretrained(self.trans_model_name)
        self.trans_model = MarianMTModel.from_pretrained(self.trans_model_name)
        if self.device.startswith("cuda"):
            self.trans_model = self.trans_model.to(self.device)
            print("Translation model moved to GPU")
            print(f"GPU memory after model loading: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")
        self.frames = []
        self.last_processed = time.time()
        self.is_speaking = False
        self.silence_start = None
        self.processing = False
        self.last_context = ""

    def is_silent(self, data):
        audio_data = np.frombuffer(data, dtype=np.int16)
        return np.abs(audio_data).mean() < SILENCE_THRESHOLD

    def translate(self, text, source_lang="auto"):
        try:
            text = remove_repetitions(text)
            inputs = self.trans_tokenizer(text, return_tensors="pt", padding=True)
            if self.device.startswith("cuda"):
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
            with torch.no_grad():
                translated = self.trans_model.generate(
                    **inputs, 
                    max_length=100,
                    num_beams=2,
                    early_stopping=True
                )
            if self.device.startswith("cuda"):
                translated = translated.cpu()
            return self.trans_tokenizer.decode(translated[0], skip_special_tokens=True)
        except Exception as e:
            print(f"Translation error: {e}")
            return text

    def save_audio(self, frames):
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
        temp_file.close()
        with wave.open(temp_file.name, 'wb') as wf:
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(self.audio.get_sample_size(FORMAT))
            wf.setframerate(RATE)
            wf.writeframes(b''.join(frames))
        return temp_file.name

    def process_audio(self):
        if not self.frames or self.processing:
            return
        self.processing = True
        temp_wav = None
        try:
            start_time = time.time()
            temp_wav = self.save_audio(self.frames)
            segments, info = self.asr_model.transcribe(
                temp_wav, 
                language=None,
                beam_size=3,
                vad_filter=True,
                vad_parameters=dict(min_silence_duration_ms=300),
                initial_prompt="This is an audio transcription.",
                word_timestamps=False
            )
            detected_text = " ".join([segment.text for segment in segments])
            detected_text = remove_repetitions(detected_text)
            transcription_time = time.time() - start_time
            if detected_text.strip():
                lang_code = info.language
                language_probability = info.language_probability
                language = language_probability > 0.5 and lang_code or "unknown"
                print(f"\n🔊 Detected [{language}] (confidence: {language_probability:.2f}, time: {transcription_time:.2f}s): {detected_text}")
                if language.lower() != "en" and language.lower() != "unknown":
                    translation_start = time.time()
                    translated_text = self.translate(detected_text, source_lang=language)
                    translation_time = time.time() - translation_start
                    print(f"📝 English ({translation_time:.2f}s): {translated_text}")
                else:
                    print(f"📝 English: {detected_text}")
                print("-" * 50)
                self.last_context = detected_text
                print("Ready for next input...")
        except Exception as e:
            print(f"Error processing audio: {e}")
        finally:
            if temp_wav and os.path.exists(temp_wav):
                try:
                    os.unlink(temp_wav)
                except:
                    pass
            self.processing = False

    def audio_callback(self, in_data, frame_count, time_info, status):
        try:
            self.frames.append(in_data)
            is_current_silent = self.is_silent(in_data)
            if is_current_silent:
                if self.is_speaking:
                    if self.silence_start is None:
                        self.silence_start = time.time()
                    elif time.time() - self.silence_start > MIN_SILENCE_TIME:
                        if not self.processing:
                            self.is_speaking = False
                            self.silence_start = None
                            if len(self.frames) > RATE * 0.5 / CHUNK:
                                processing_thread = threading.Thread(target=self.process_audio)
                                processing_thread.daemon = True
                                processing_thread.start()
                            self.frames = []
                            self.last_context = ""
            else:
                self.is_speaking = True
                self.silence_start = None
                buffer_seconds = len(self.frames) * CHUNK / RATE
                if buffer_seconds > MAX_BUFFER_LENGTH:
                    if not self.processing:
                        processing_thread = threading.Thread(target=self.process_audio)
                        processing_thread.daemon = True
                        processing_thread.start()
                        recent_frames_count = int(1 * RATE / CHUNK)
                        self.frames = self.frames[-recent_frames_count:] if len(self.frames) > recent_frames_count else []
        except Exception as e:
            print(f"Error in audio callback: {e}")
        return (in_data, pyaudio.paContinue)

    def start(self):
        try:
            self.stream = self.audio.open(
                format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK,
                stream_callback=self.audio_callback
            )
            print("\n==== REAL-TIME TRANSLATION SYSTEM ====")
            print("Repetition prevention: Enabled")
            print("=" * 40)
            print("🎤 Listening... Speak now! (Press Ctrl+C to exit)")
            self.stream.start_stream()
            try:
                while self.stream.is_active():
                    time.sleep(0.1)
            except KeyboardInterrupt:
                print("\nExiting...")
            except Exception as e:
                print(f"Error during streaming: {e}")
        except Exception as e:
            print(f"Error starting audio stream: {e}")
        finally:
            self.stop()

    def stop(self):
        try:
            if self.stream:
                self.stream.stop_stream()
                self.stream.close()
            self.audio.terminate()
            print("Stopped recording.")
        except Exception as e:
            print(f"Error stopping translator: {e}")

if __name__ == "__main__":
    try:
        translator = RealTimeTranslator(whisper_model="large", use_gpu=True)
        translator.start()
    except Exception as e:
        print(f"Error in main: {e}")