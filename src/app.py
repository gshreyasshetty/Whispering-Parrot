import streamlit as st
import os
import numpy as np
import tempfile
import time
import matplotlib.pyplot as plt
from audio_recorder_streamlit import audio_recorder
import wave
import threading
import sys

# Disable file watcher for torch to avoid the terminal errors
os.environ["STREAMLIT_WATCH_MODULES"] = "false"

# Import after setting environment variables
from faster_whisper import WhisperModel

st.set_page_config(
    page_title="Whispering-Parrot - Speech Translator",
    page_icon="🦜",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS with dark text color for results
st.markdown("""
<style>
    .main-header {
        font-size: 36px;
        font-weight: bold;
        color: #00897B;
        text-align: center;
        margin-bottom: 20px;
    }
    .result-area {
        background-color: #F5F5F5;
        padding: 20px;
        border-radius: 10px;
        min-height: 100px;
        color: #333333; /* Dark text color for visibility */
        font-size: 18px;
    }
    .language-tag {
        color: #555555;
        font-weight: bold;
        margin-bottom: 10px;
    }
    /* Reduce padding for better space usage */
    .st-emotion-cache-16txtl3 {
        padding-top: 1rem;
        padding-bottom: 1rem;
    }
    div[data-testid="stStatusWidget"] {
        display: none;
    }
    .stButton button {
        background-color: #00897B;
        color: white;
        border: none;
    }
    .stButton button:hover {
        background-color: #00695C;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# Cache the model loading for better performance
@st.cache_resource
def load_whisper_model(model_size, device="cpu"):
    """Load and cache the Whisper model"""
    compute_type = "int8" if device == "cpu" else "float16"
    try:
        model = WhisperModel(
            model_size, 
            device=device,
            compute_type=compute_type,
            download_root="./models"
        )
        return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

def save_audio_to_wav(audio_data, sample_rate=16000):
    """Save audio data to a WAV file."""
    with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp_audio:
        temp_path = temp_audio.name
    
    with wave.open(temp_path, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(audio_data.tobytes())
    
    return temp_path

def display_audio_visualization(audio_data):
    """Display a waveform visualization of audio data."""
    # Using a smaller figure size
    fig, ax = plt.subplots(figsize=(7, 1))
    ax.plot(audio_data, color='#00897B', alpha=0.7)
    ax.set_xlim(0, len(audio_data))
    ax.set_ylim(-1, 1)
    ax.axis('off')
    st.pyplot(fig)

def process_audio_file(audio_path, model, progress_bar=None):
    """Process audio file using faster-whisper"""
    start_time = time.time()
    
    # Update progress
    if progress_bar:
        progress_bar.progress(0.2, text="Starting transcription...")
    
    # Transcribe the audio
    segments, info = model.transcribe(
        audio_path, 
        language=None,
        beam_size=3,
        vad_filter=True,
        vad_parameters=dict(min_silence_duration_ms=300),
        initial_prompt="This is an audio transcription.",
        word_timestamps=False
    )
    
    if progress_bar:
        progress_bar.progress(0.7, text="Processing transcription results...")
    
    # Get the results
    detected_text = " ".join([segment.text for segment in segments])
    lang_code = info.language
    language_probability = info.language_probability
    
    # Calculate processing time
    process_time = time.time() - start_time
    
    if progress_bar:
        progress_bar.progress(1.0, text=f"Done in {process_time:.2f}s")
    
    return detected_text, lang_code, language_probability, process_time

def translate_text(text, lang_code, model):
    """Placeholder for translation - add real translation here"""
    # This would normally use your MarianMT model from real_time_translation
    # For now, we'll just return the text for non-English inputs
    if lang_code != 'en' and lang_code.lower() != 'unknown':
        return f"[Translation would appear here - requires MarianMT model]"
    return text

def main():
    st.markdown('<div class="main-header">🦜 Whispering-Parrot: Multilingual Speech Translator</div>', unsafe_allow_html=True)
    
    with st.sidebar:
        st.image("https://img.icons8.com/color/96/000000/parrot.png", width=100)
        st.markdown("## Settings")
        
        model_size = st.selectbox(
            "Whisper Model Size", 
            ["tiny", "base", "small", "medium", "large"],  # Added large model
            index=2,
            help="Larger models are more accurate but slower"
        )
        
        # Add real-time mode option
        real_time_mode = st.checkbox("Real-time processing", value=True, 
                                  help="Process audio immediately after recording")
        
        # GPU option if available
        use_gpu = False
        try:
            import torch
            if torch.cuda.is_available():
                use_gpu = st.checkbox("Use GPU (faster processing)", value=True)
                if use_gpu:
                    st.success(f"✅ Using GPU: {torch.cuda.get_device_name(0)}")
        except:
            pass
        
        st.info(
            "Whispering-Parrot uses Whisper for multilingual speech recognition. "
            "Larger models are more accurate but slower."
        )
    
    # Load the model (using cache)
    device = "cuda" if use_gpu else "cpu"
    with st.spinner("Loading model..."):
        model = load_whisper_model(model_size, device)
    
    # Main content
    tab1, tab2 = st.tabs(["Record Audio", "Upload Audio"])
    
    with tab1:
        col1, col2 = st.columns([2, 1])
        with col1:
            st.markdown("### Record Your Voice")
            st.markdown("Click the button below and speak into your microphone")
        
        with col2:
            # Add a rerun button for convenience
            if st.button("🔄 Reset", help="Clear current recording and start fresh"):
                st.rerun()
        
        audio_bytes = audio_recorder(
            text="Click to record",
            recording_color="#e74c3c",
            neutral_color="#00897B",
            key="audio_recorder"
        )
        
        if audio_bytes:
            # Convert audio_bytes to numpy array for visualization
            audio_array = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
            
            # Smaller waveform
            display_audio_visualization(audio_array)
            
            # Save audio to temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp_audio:
                temp_audio.write(audio_bytes)
                temp_path = temp_audio.name
            
            # Process audio with progress bar
            progress_bar = st.progress(0, text="Preparing to process audio...")
            
            try:
                detected_text, lang_code, language_probability, process_time = process_audio_file(
                    temp_path, model, progress_bar
                )
                
                language_name = {
                    'en': 'English', 'es': 'Spanish', 'fr': 'French', 'de': 'German', 
                    'it': 'Italian', 'pt': 'Portuguese', 'nl': 'Dutch', 'ru': 'Russian',
                    'zh': 'Chinese', 'ja': 'Japanese', 'ko': 'Korean', 'ar': 'Arabic',
                    'hi': 'Hindi', 'bn': 'Bengali', 'ur': 'Urdu', 'te': 'Telugu',
                    'ta': 'Tamil', 'mr': 'Marathi', 'gu': 'Gujarati', 'kn': 'Kannada'
                }.get(lang_code, lang_code)
                
                # Display results in card-like containers
                st.markdown(f"<div class='language-tag'>Detected: {language_name} (confidence: {language_probability:.2f}) • Processed in {process_time:.2f}s</div>", unsafe_allow_html=True)
                
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("### Transcription:")
                    st.markdown(f"<div class='result-area'>{detected_text}</div>", unsafe_allow_html=True)
                
                with col2:
                    if lang_code.lower() != 'en' and lang_code.lower() != 'unknown':
                        st.markdown("### English Translation:")
                        translated_text = translate_text(detected_text, lang_code, model)
                        st.markdown(f"<div class='result-area'>{translated_text}</div>", unsafe_allow_html=True)
                
                # Clean up temp file
                try:
                    os.unlink(temp_path)
                except:
                    pass
                
            except Exception as e:
                st.error(f"Error processing audio: {str(e)}")
    
    with tab2:
        st.markdown("### Upload Audio File")
        uploaded_file = st.file_uploader("Choose an audio file", type=['wav', 'mp3', 'ogg'])
        
        if uploaded_file:
            # Display audio player
            st.audio(uploaded_file)
            
            # Save uploaded file to temp location
            with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp_audio:
                temp_audio.write(uploaded_file.getvalue())
                temp_path = temp_audio.name
            
            # Process with progress bar
            progress_bar = st.progress(0, text="Preparing to process audio...")
            
            try:
                detected_text, lang_code, language_probability, process_time = process_audio_file(
                    temp_path, model, progress_bar
                )
                
                language_name = {
                    'en': 'English', 'es': 'Spanish', 'fr': 'French', 'de': 'German', 
                    'it': 'Italian', 'pt': 'Portuguese', 'nl': 'Dutch', 'ru': 'Russian',
                    'zh': 'Chinese', 'ja': 'Japanese', 'ko': 'Korean', 'ar': 'Arabic',
                    'hi': 'Hindi', 'bn': 'Bengali', 'ur': 'Urdu', 'te': 'Telugu',
                    'ta': 'Tamil', 'mr': 'Marathi', 'gu': 'Gujarati', 'kn': 'Kannada'
                }.get(lang_code, lang_code)
                
                # Display results in card-like containers
                st.markdown(f"<div class='language-tag'>Detected: {language_name} (confidence: {language_probability:.2f}) • Processed in {process_time:.2f}s</div>", unsafe_allow_html=True)
                
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("### Transcription:")
                    st.markdown(f"<div class='result-area'>{detected_text}</div>", unsafe_allow_html=True)
                
                with col2:
                    if lang_code.lower() != 'en' and lang_code.lower() != 'unknown':
                        st.markdown("### English Translation:")
                        translated_text = translate_text(detected_text, lang_code, model)
                        st.markdown(f"<div class='result-area'>{translated_text}</div>", unsafe_allow_html=True)
                
                # Clean up temp file
                try:
                    os.unlink(temp_path)
                except:
                    pass
                
            except Exception as e:
                st.error(f"Error processing audio: {str(e)}")
    
    # Footer
    st.markdown("---")
    st.markdown("### 📌 Key Features")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("- 🌍 **Multilingual Support**")
        st.markdown("- 🎯 **High Accuracy**")
    with col2:
        st.markdown("- 🔊 **Record or Upload**")
        st.markdown("- ⚡ **GPU Acceleration**")
    with col3:
        st.markdown("- ⏱️ **Real-time Processing**")
        st.markdown("- 🔄 **Easy Reset**")
    
    st.markdown("""<div style="text-align:center; margin-top:30px; color:#666;">
    Whispering-Parrot © 2025 | Powered by OpenAI Whisper</div>""", unsafe_allow_html=True)

if __name__ == "__main__":
    main()