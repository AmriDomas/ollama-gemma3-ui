import speech_recognition as sr
import pydub
from pydub import AudioSegment
import tempfile
import os
from datetime import datetime
import io
import time

class AudioProcessor:
    def __init__(self):
        self.recognizer = sr.Recognizer()
        self.history = []
        self.is_recording = False
        
    def start_recording(self, duration=5):
        """Start recording audio"""
        self.is_recording = True
        return self.record_audio(duration)
    
    def stop_recording(self):
        """Stop recording"""
        self.is_recording = False
    
    def record_audio(self, duration=5):
        """Record audio from microphone"""
        try:
            with sr.Microphone() as source:
                # Adjust for ambient noise
                self.recognizer.adjust_for_ambient_noise(source, duration=1)
                
                # Record
                st.info(f"🎤 Recording for {duration} seconds...")
                audio = self.recognizer.listen(source, timeout=duration, phrase_time_limit=duration)
                
                # Save to temporary file
                with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
                    with open(tmp.name, 'wb') as f:
                        f.write(audio.get_wav_data())
                    return tmp.name
                    
        except Exception as e:
            return f"Recording error: {str(e)}"
    
    def transcribe_audio(self, audio_file):
        """Transcribe audio file to text"""
        try:
            if isinstance(audio_file, str) and os.path.exists(audio_file):
                with sr.AudioFile(audio_file) as source:
                    audio = self.recognizer.record(source)
                    text = self.recognizer.recognize_google(audio, language='id-ID')
                    
                    # Save to history
                    self.history.append({
                        'timestamp': datetime.now(),
                        'text': text,
                        'path': audio_file
                    })
                    
                    return text
            else:
                return "Audio file not found"
                
        except sr.UnknownValueError:
            return "Could not understand audio"
        except sr.RequestError as e:
            return f"API error: {str(e)}"
        except Exception as e:
            return f"Transcription error: {str(e)}"
    
    def text_to_speech(self, text, lang='id'):
        """Convert text to speech"""
        try:
            from gtts import gTTS
            import base64
            
            tts = gTTS(text=text, lang=lang, slow=False)
            
            with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as tmp:
                tts.save(tmp.name)
                
                # Read file and encode to base64 for HTML audio
                with open(tmp.name, 'rb') as f:
                    audio_bytes = f.read()
                
                # Return as base64 for HTML audio tag
                audio_base64 = base64.b64encode(audio_bytes).decode()
                audio_html = f'<audio autoplay controls><source src="data:audio/mp3;base64,{audio_base64}" type="audio/mp3"></audio>'
                
                return audio_html
                
        except ImportError:
            return "Install gTTS: pip install gtts"
        except Exception as e:
            return f"TTS error: {str(e)}"
    
    def play_last_recording(self):
        """Play the last recording"""
        if self.history:
            last_recording = self.history[-1]['path']
            if os.path.exists(last_recording):
                with open(last_recording, 'rb') as f:
                    audio_bytes = f.read()
                return audio_bytes
        return None