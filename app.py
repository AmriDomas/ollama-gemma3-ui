import streamlit as st
import ollama
import os
import tempfile
from datetime import datetime
import pandas as pd
import json
import base64
from typing import List, Dict, Optional
import time
from PIL import Image
import io
import requests
from streamlit_autorefresh import st_autorefresh
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

# Import custom modules
try:
    from audio_handler import AudioProcessor
    from image_processor import ImageAnalyzer
    from collaborative import CollaborationSession
    AUDIO_ENABLED = True
    IMAGE_ENABLED = True
    COLLAB_ENABLED = True
except ImportError:
    AUDIO_ENABLED = False
    IMAGE_ENABLED = False
    COLLAB_ENABLED = False
    st.warning("Some features disabled. Install extra packages.")

# ==================== CONFIGURATION ====================
st.set_page_config(
    page_title="Gemma 3 4B Ultimate UI",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/yourrepo',
        'Report a bug': 'https://github.com/yourrepo/issues',
        'About': "# Gemma 3 4B Ultimate Chat UI"
    }
)

# ==================== CUSTOM CSS & JS ====================
st.markdown("""
<style>
    /* Main styling */
    .main-header {
        font-size: 3rem !important;
        background: linear-gradient(45deg, #FF6B6B, #4ECDC4, #45B7D1);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    /* Chat bubbles */
    .user-message {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 15px;
        border-radius: 20px 20px 5px 20px;
        margin: 10px 0;
        max-width: 80%;
        margin-left: auto;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    
    .ai-message {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
        padding: 15px;
        border-radius: 20px 20px 20px 5px;
        margin: 10px 0;
        max-width: 80%;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    
    /* Cards */
    .feature-card {
        background: rgba(255,255,255,0.1);
        border-radius: 15px;
        padding: 20px;
        margin: 10px 0;
        border-left: 5px solid #4ECDC4;
        transition: transform 0.3s;
    }
    
    .feature-card:hover {
        transform: translateY(-5px);
        background: rgba(255,255,255,0.15);
    }
    
    /* Buttons */
    .stButton > button {
        background: linear-gradient(45deg, #6a11cb 0%, #2575fc 100%);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 10px 20px;
        font-weight: bold;
        transition: all 0.3s;
    }
    
    .stButton > button:hover {
        transform: scale(1.05);
        box-shadow: 0 5px 15px rgba(0,0,0,0.3);
    }
    
    /* Progress bars */
    .stProgress > div > div > div > div {
        background: linear-gradient(45deg, #FF6B6B, #4ECDC4);
    }
    
    /* Sidebar */
    .css-1d391kg {
        background: linear-gradient(180deg, #0f0c29, #302b63, #24243e);
    }
</style>

<script>
// Audio notification for new messages
function playNotificationSound() {
    const audio = new Audio('data:audio/wav;base64,UDk0FQAAAAAA');
    audio.volume = 0.3;
    audio.play();
}

// Auto-scroll to bottom
function scrollToBottom() {
    window.scrollTo(0, document.body.scrollHeight);
}

// Typing indicator animation
const typingIndicator = document.createElement('div');
typingIndicator.innerHTML = '<div class="typing"><span></span><span></span><span></span></div>';
</script>
""", unsafe_allow_html=True)

# ==================== SESSION STATE ====================
def cleanup_chat_history():
    """Convert all datetime objects to strings in chat history"""
    if st.session_state.chat_history:
        cleaned = []
        for item in st.session_state.chat_history:
            if isinstance(item, dict) and 'timestamp' in item:
                if isinstance(item['timestamp'], datetime):
                    item_copy = item.copy()
                    item_copy['timestamp'] = item['timestamp'].strftime("%Y-%m-%d %H:%M:%S")
                    cleaned.append(item_copy)
                else:
                    cleaned.append(item)
            else:
                cleaned.append(item)
        st.session_state.chat_history = cleaned

# Panggil fungsi cleanup di init_session_state:
def init_session_state():
    """Initialize all session state variables"""
    defaults = {
        "messages": [],
        "model": "gemma3:4b",
        "temperature": 0.7,
        "rag_enabled": False,
        "voice_enabled": False,
        "image_analysis": False,
        "collab_session": CollaborationSession() if COLLAB_ENABLED else None,
        "current_tab": "chat",
        "api_key": "",
        "theme": "dark",
        "auto_refresh": False,
        "tts_enabled": False,
        "streaming_speed": "medium",
        "selected_plugins": [],
        "chat_history": [],
        "favorites": [],
        "workspaces": {"default": []},
        "current_workspace": "default",
        "notifications": []
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value
    
    # Cleanup existing data
    cleanup_chat_history()

init_session_state()

def get_plugin_icon(plugin: str) -> str:
        """Get icon for plugin"""
        icons = {
            "email": "📧",
            "code": "💻",
            "data": "📊",
            "creative": "🎨",
            "research": "🔍",
            "translate": "🌐"
        }
        return icons.get(plugin, "🧩")

def email_writer_ui():
    """Email writing plugin UI"""
    st.write("✉️ Email Writer")
    recipient = st.text_input("Recipient", key="email_recipient")
    subject = st.text_input("Subject", key="email_subject")
    tone = st.selectbox("Tone", ["Formal", "Casual", "Friendly", "Professional"], key="email_tone")
    key_points = st.text_area("Key Points", key="email_points")
    
    if st.button("Generate Email", key="email_generate"):
        prompt = f"Write a {tone.lower()} email to {recipient} about: {subject}. Key points: {key_points}"
        try:
            response = ollama.chat(
                model=st.session_state.model,
                messages=[{"role": "user", "content": prompt}]
            )
            st.text_area("Generated Email", response['message']['content'], height=200, key="email_output")
        except Exception as e:
            st.error(f"Failed to generate email: {str(e)}")

def code_assistant_ui():
    """Code assistant plugin UI"""
    st.write("💻 Code Assistant")
    language = st.selectbox("Language", ["Python", "JavaScript", "Java", "C++", "Go", "Rust"], key="code_lang")
    task = st.selectbox("Task", ["Write function", "Debug", "Optimize", "Explain", "Convert"], key="code_task")
    code_input = st.text_area("Your code", height=150, key="code_input")
    
    if st.button("Get Help", key="code_help"):
        prompt = f"As a {language} expert, {task} this code:\n\n{code_input}"
        try:
            response = ollama.chat(
                model=st.session_state.model,
                messages=[{"role": "user", "content": prompt}]
            )
            st.code(response['message']['content'], language=language.lower())
        except Exception as e:
            st.error(f"Failed to analyze code: {str(e)}")

def data_analyzer_ui():
    """Data analyzer plugin UI"""
    st.write("📊 Data Analyzer")
    data_input = st.text_area("Paste your data (CSV/JSON) or describe", height=150, key="data_input")
    analysis_type = st.selectbox("Analysis Type", ["Summarize", "Find patterns", "Predict trends", "Visualize"], key="data_analysis")
    
    if st.button("Analyze", key="data_analyze"):
        prompt = f"Analyze this data ({analysis_type}): {data_input}"
        try:
            response = ollama.chat(
                model=st.session_state.model,
                messages=[{"role": "user", "content": prompt}]
            )
            st.write(response['message']['content'])
        except Exception as e:
            st.error(f"Failed to analyze data: {str(e)}")

def creative_writer_ui():
    """Creative writer plugin UI"""
    st.write("🎨 Creative Writer")
    genre = st.selectbox("Genre", ["Story", "Poem", "Script", "Article", "Song"], key="creative_genre")
    theme = st.text_input("Theme/Topic", key="creative_theme")
    length = st.slider("Length (words)", 50, 500, 150, key="creative_length")
    
    if st.button("Create", key="creative_create"):
        prompt = f"Write a {genre.lower()} about '{theme}' with about {length} words"
        try:
            response = ollama.chat(
                model=st.session_state.model,
                messages=[{"role": "user", "content": prompt}]
            )
            st.text_area("Result", response['message']['content'], height=200, key="creative_output")
        except Exception as e:
            st.error(f"Failed to generate content: {str(e)}")

def send_message(prompt: str):
    """Send message to Ollama"""
    if not prompt.strip():
        return
    
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    try:
        # Prepare messages for Ollama
        messages_for_ollama = []
        for msg in st.session_state.messages[-10:]:
            messages_for_ollama.append({"role": msg["role"], "content": msg["content"]})
        
        # Get response
        start_time = time.time()
        response = ollama.chat(
            model=st.session_state.model,
            messages=messages_for_ollama,
            options={"temperature": st.session_state.temperature}
        )
        response_time = time.time() - start_time
        
        ai_response = response['message']['content']
        
        # Add assistant response
        st.session_state.messages.append({
            "role": "assistant", 
            "content": ai_response
        })
        
        # Simpan ke chat_history dengan KOLOM YANG KONSISTEN
        st.session_state.chat_history.append({
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "model": st.session_state.model,
            "user": prompt,  # KOLOM: 'user' bukan 'user_prompt'
            "assistant": ai_response,
            "response_length": len(ai_response),
            "response_time": round(response_time, 2)
        })
        
        st.rerun()  # Refresh untuk menampilkan pesan baru
        
    except Exception as e:
        st.error(f"Error: {str(e)}")
        st.session_state.messages.append({
            "role": "assistant", 
            "content": f"Sorry, I encountered an error: {str(e)}"
        })

def export_chat():
    """Export chat history"""
    if st.session_state.chat_history:
        # Pastikan semua timestamp sudah string
        cleaned_history = []
        for item in st.session_state.chat_history:
            if isinstance(item.get('timestamp'), datetime):
                # Convert datetime to string
                cleaned_item = item.copy()
                cleaned_item['timestamp'] = item['timestamp'].strftime("%Y-%m-%d %H:%M:%S")
                cleaned_history.append(cleaned_item)
            else:
                cleaned_history.append(item)
        
        df = pd.DataFrame(cleaned_history)
        csv = df.to_csv(index=False)
        
        st.download_button(
            "📥 Download Chat History",
            csv,
            file_name=f"chat_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )
    else:
        st.info("No chat history to export")

def regenerate_message(message_index: int):
    """Simple message regeneration"""
    try:
        # Find the user message before this AI response
        if message_index > 0:
            # Remove the AI message
            old_message = st.session_state.messages.pop(message_index)
            
            # Get the user message that prompted it
            user_message = st.session_state.messages[message_index-1]["content"]
            
            # Regenerate with same prompt
            with st.spinner("Regenerating response..."):
                response = ollama.chat(
                    model=st.session_state.model,
                    messages=[{"role": "user", "content": user_message}],
                    options={"temperature": st.session_state.temperature}
                )
                
                # Add new response
                st.session_state.messages.insert(message_index, {
                    "role": "assistant",
                    "content": response['message']['content']
                })
                
            st.rerun()
            
    except Exception as e:
        st.error(f"Failed to regenerate: {str(e)}")

def voice_input():
    """Handle voice input"""
    if not AUDIO_ENABLED:
        st.warning("Audio features disabled. Install: pip install speechrecognition pydub")
        return
    
    # Buat placeholder untuk recording
    recording_placeholder = st.empty()
    
    with recording_placeholder.container():
        st.info("🎤 Recording... Speak now (5 seconds)")
        
        # Record audio
        try:
            # Initialize audio processor jika belum ada
            if 'audio_processor' not in st.session_state:
                st.session_state.audio_processor = AudioProcessor()
            
            # Record audio
            audio_file = st.session_state.audio_processor.record_audio(duration=5)
            
            if audio_file:
                # Transcribe
                text = st.session_state.audio_processor.transcribe_audio(audio_file)
                
                if text and "error" not in text.lower():
                    # Tambah ke chat
                    st.session_state.messages.append({
                        "role": "user",
                        "content": f"🎤 {text}"
                    })
                    
                    # Kirim ke AI
                    send_message(text)
                    st.success("Voice message sent!")
                else:
                    st.error("Could not transcribe audio. Please try again.")
        
        except Exception as e:
            st.error(f"Voice input error: {str(e)}")

def file_upload():
    """Handle file upload"""
    # Buat file uploader di modal/popup
    uploaded_file = st.file_uploader(
        "Choose a file to upload",
        type=['txt', 'pdf', 'docx', 'md', 'py', 'js', 'html', 'css', 'json', 'csv', 'jpg', 'png', 'jpeg'],
        key="file_upload_popup",
        help="Upload file to analyze or attach to chat"
    )
    
    if uploaded_file:
        file_details = {
            "filename": uploaded_file.name,
            "filetype": uploaded_file.type,
            "filesize": f"{uploaded_file.size / 1024:.2f} KB"
        }
        
        st.success(f"✅ Uploaded: {uploaded_file.name}")
        
        # Tampilkan preview berdasarkan tipe file
        if uploaded_file.type == "text/plain" or uploaded_file.name.endswith('.txt'):
            content = uploaded_file.read().decode('utf-8')
            st.text_area("File Content", content[:1000] + ("..." if len(content) > 1000 else ""), height=200)
            
            # Tambah ke chat
            if st.button("📝 Send to Chat"):
                st.session_state.messages.append({
                    "role": "user",
                    "content": f"📎 File: {uploaded_file.name}\n\nContent:\n{content[:500]}..."
                })
                send_message(f"Analyze this file content: {content[:500]}")
        
        elif uploaded_file.type.startswith('image/'):
            from PIL import Image
            image = Image.open(uploaded_file)
            st.image(image, caption=uploaded_file.name, use_column_width=True)
            
            # Tambah ke chat
            if st.button("🖼️ Send to Chat"):
                st.session_state.messages.append({
                    "role": "user",
                    "content": f"🖼️ Image: {uploaded_file.name} (Image uploaded)"
                })
                st.info("Image uploaded. Use Vision tab for analysis.")
        
        elif uploaded_file.type in ["application/json", "text/csv"]:
            try:
                content = uploaded_file.read().decode('utf-8')
                st.text_area("File Content", content[:500] + ("..." if len(content) > 500 else ""), height=150)
                
                if st.button("📊 Analyze Data"):
                    st.session_state.messages.append({
                        "role": "user",
                        "content": f"📊 Data file: {uploaded_file.name}\n\nData preview:\n{content[:300]}..."
                    })
                    send_message(f"Analyze this data from {uploaded_file.name}: {content[:300]}")
            except:
                st.warning("Could not read file content")
        
        else:
            st.info(f"File type: {uploaded_file.type}")
            if st.button("📎 Attach to Chat"):
                st.session_state.messages.append({
                    "role": "user",
                    "content": f"📎 Attachment: {uploaded_file.name} ({file_details['filesize']})"
                })
                st.success("File attached to chat!")

def get_random_prompt():
    """Get random prompt for inspiration"""
    prompts = [
        "Explain quantum computing like I'm 10 years old",
        "Write a short poem about artificial intelligence",
        "What are the ethical implications of AI?",
        "Create a workout plan for beginners",
        "Explain blockchain technology in simple terms",
        "Write a recipe for chocolate chip cookies",
        "What's the future of renewable energy?",
        "Create a business plan for a coffee shop",
        "Explain how neural networks work",
        "Write a short story about time travel",
        "Apa ibu kota Indonesia?",
        "Jelaskan sejarah kemerdekaan Indonesia",
        "Apa saja makanan khas Indonesia?",
        "Buatkan rencana perjalanan 3 hari ke Bali",
        "Bagaimana cara membuat website sederhana?",
        "Jelaskan sistem pemerintahan di Indonesia",
        "Apa itu machine learning?",
        "Buatkan ide startup untuk mahasiswa",
        "Jelaskan tentang perubahan iklim",
        "Bagaimana cara investasi untuk pemula?"
    ]
    import random
    return random.choice(prompts)

def share_message(message):
    """Share message via various methods"""
    share_options = st.selectbox(
        "Share via",
        ["Copy to clipboard", "Save as text", "Share link", "Export as image"],
        key="share_options"
    )
    
    if share_options == "Copy to clipboard":
        import pyperclip
        try:
            pyperclip.copy(message["content"])
            st.success("Copied to clipboard!")
        except:
            st.code(message["content"])
            st.info("Manual copy: Select text above and copy")
    
    elif share_options == "Save as text":
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"message_{timestamp}.txt"
        
        st.download_button(
            "📥 Download",
            message["content"],
            file_name=filename,
            mime="text/plain"
        )
    
    elif share_options == "Share link":
        st.info("Link sharing requires backend setup")
        st.code("https://your-app.com/share/12345")  # Placeholder
    
    elif share_options == "Export as image":
        st.info("Image export requires additional packages")
        st.warning("Install: pip install pillow")

def show_templates():
    """Show prompt templates"""
    templates = {
        "\U0001f4e7 Email": "Write a professional email about [topic]",
        "\U0001f4dd Report": "Create a report on [subject] with sections for introduction, analysis, and conclusion",
        "\U0001f4a1 Brainstorm": "Brainstorm ideas for [project]",
        "\U0001f4da Summary": "Summarize the key points of [text/topic]",
        "\U0001f50d Research": "Research about [topic] and provide sources",
        "\U0001f3a8 Creative": "Write a creative story about [theme]",
        "\U0001f4bb Code": "Write Python code to [task]",
        "\U0001f4ca Analysis": "Analyze [data/topic] and provide insights"
    }
    
    selected = st.selectbox("Choose template", list(templates.keys()))
    template_text = st.text_area("Template", templates[selected], height=100)
    
    if st.button("Use Template"):
        st.session_state.messages.append({
            "role": "user",
            "content": template_text
        })
        st.success("Template added to chat!")

def restart_session():
    """Restart the session"""
    if st.button("Confirm Restart", type="primary"):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()

def clear_all_data():
    """Clear all user data"""
    st.warning("This will delete ALL your data!")
    confirm = st.checkbox("I understand this cannot be undone")
    
    if confirm and st.button("Delete All Data", type="primary"):
        keys_to_keep = ['_last_run', '_widget_state']
        for key in [k for k in st.session_state.keys() if k not in keys_to_keep]:
            del st.session_state[key]
        st.success("All data deleted!")
        time.sleep(2)
        st.rerun()

def preview_theme(theme_name):
    """Preview theme changes"""
    
    themes = {
        "Dark": {
            "primary": "#0E1117",
            "secondary": "#262730",
            "text": "#FAFAFA"
        },
        "Light": {
            "primary": "#FFFFFF",
            "secondary": "#F0F2F6",
            "text": "#262730"
        },
        "Blue": {
            "primary": "#1E3A8A",
            "secondary": "#3B82F6",
            "text": "#FFFFFF"
        },
        "Green": {
            "primary": "#065F46",
            "secondary": "#10B981",
            "text": "#FFFFFF"
        },
        "Purple": {
            "primary": "#5B21B6",
            "secondary": "#8B5CF6",
            "text": "#FFFFFF"
        }
    }
    
    if theme_name in themes:
        theme = themes[theme_name]
        
        st.info(f"Previewing {theme_name} Theme")
        
        # Preview container
        with st.container(border=True):
            st.markdown(f"""
            <div style="
                background-color: {theme['primary']};
                color: {theme['text']};
                padding: 20px;
                border-radius: 10px;
                margin: 10px 0;
            ">
                <h3 style="color: {theme['text']};">{theme_name} Theme Preview</h3>
                <p>This is how the {theme_name.lower()} theme will look.</p>
                
                <div style="
                    background-color: {theme['secondary']};
                    padding: 15px;
                    border-radius: 5px;
                    margin: 10px 0;
                ">
                    <p style="color: {theme['text']};">Sample message bubble</p>
                </div>
                
                <button style="
                    background-color: {theme['secondary']};
                    color: {theme['text']};
                    border: none;
                    padding: 10px 20px;
                    border-radius: 5px;
                    cursor: pointer;
                ">
                    Sample Button
                </button>
            </div>
            """, unsafe_allow_html=True)
        
        # Apply theme button
        if st.button(f"Apply {theme_name} Theme"):
            st.session_state.theme = theme_name.lower()
            st.success(f"{theme_name} theme applied! Refresh page to see full changes.")
            
            # Update CSS
            update_css_for_theme(theme_name)
    
    elif theme_name == "Custom":
        st.info("Custom theme settings saved. Full preview requires app refresh.")

def update_css_for_theme(theme_name):
    """Update CSS based on selected theme"""
    
    theme_css = {
        "dark": """
            .stApp {
                background-color: #0E1117;
                color: #FAFAFA;
            }
            .css-1d391kg {
                background: linear-gradient(180deg, #0f0c29, #302b63, #24243e);
            }
        """,
        "light": """
            .stApp {
                background-color: #FFFFFF;
                color: #262730;
            }
            .css-1d391kg {
                background: linear-gradient(180deg, #f8f9fa, #e9ecef, #dee2e6);
            }
        """,
        "blue": """
            .stApp {
                background-color: #1E3A8A;
                color: #FFFFFF;
            }
            .css-1d391kg {
                background: linear-gradient(180deg, #1e40af, #3b82f6, #60a5fa);
            }
        """,
        "green": """
            .stApp {
                background-color: #065F46;
                color: #FFFFFF;
            }
            .css-1d391kg {
                background: linear-gradient(180deg, #047857, #10b981, #34d399);
            }
        """,
        "purple": """
            .stApp {
                background-color: #5B21B6;
                color: #FFFFFF;
            }
            .css-1d391kg {
                background: linear-gradient(180deg, #5b21b6, #7c3aed, #8b5cf6);
            }
        """
    }
    
    if theme_name.lower() in theme_css:
        st.markdown(f"<style>{theme_css[theme_name.lower()]}</style>", unsafe_allow_html=True)

def test_api_connections():
    """Test API connections"""
    
    st.info("Testing API connections...")
    
    results = []
    
    # Test Ollama connection
    try:
        ollama.list()
        results.append(("\u2705 Ollama", "Connected"))
    except:
        results.append(("\u274c Ollama", "Not connected"))
    
    # Test other APIs (placeholder)
    if st.session_state.get('openai_key'):
        results.append(("\u2705 OpenAI", "API key set"))
    else:
        results.append(("\u26a0\ufe0f OpenAI", "No API key"))
    
    if st.session_state.get('google_ai_key'):
        results.append(("\u2705 Google AI", "API key set"))
    else:
        results.append(("\u26a0\ufe0f Google AI", "No API key"))
    
    # Display results
    for service, status in results:
        st.write(f"{service}: {status}")

def reset_to_defaults():
    """Reset settings to defaults"""
    
    defaults = {
        "temperature": 0.7,
        "model": "gemma3:4b",
        "rag_enabled": False,
        "voice_enabled": False,
        "image_analysis": False,
        "tts_enabled": False,
        "streaming_speed": "medium",
        "selected_plugins": ["code", "email"],
        "theme": "dark"
    }
    
    for key, value in defaults.items():
        st.session_state[key] = value
    
    st.success("Settings reset to defaults!")

def factory_reset():
    """Factory reset the application"""
    
    st.error("\u26a0\ufe0f FACTORY RESET")
    st.warning("This will delete ALL settings and data!")
    
    confirm = st.checkbox("I understand this will delete everything")
    confirm2 = st.checkbox("I have backed up important data")
    
    if confirm and confirm2 and st.button("CONFIRM FACTORY RESET", type="primary"):
        # Get list of keys to keep
        keys_to_keep = ['_last_run']
        
        # Delete all other keys
        for key in list(st.session_state.keys()):
            if key not in keys_to_keep:
                del st.session_state[key]
        
        st.success("Factory reset complete! Refreshing...")
        time.sleep(2)
        st.rerun()

# Jika ada fungsi lain yang belum didefinisikan, tambahkan placeholder:

def describe_image_with_ai(image):
    """Use AI to describe image"""
    try:
        # Simulate AI description
        return "This is an image. To get detailed description, install a vision model like LLaVA."
    except:
        return "Image description not available."

def show_voting_interface():
    """Show voting interface for collaboration"""
    st.info("Voting feature requires real-time collaboration setup")

def private_message(user_id: str):
    """Send private message in collaboration"""
    st.info(f"Private messaging to {user_id} requires WebSocket setup")


def show_document_preview(file_content, filename):
    """Show document preview in modal"""
    
    with st.expander(f"📄 Preview: {filename}", expanded=True):
        # Display based on content type
        if len(file_content) > 10000:
            st.warning(f"Document too large ({len(file_content)} chars). Showing first 10,000 characters.")
            st.text_area("Content", file_content[:10000], height=300)
        else:
            st.text_area("Content", file_content, height=400)
        
        # Statistics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Characters", len(file_content))
        with col2:
            words = len(file_content.split())
            st.metric("Words", words)
        with col3:
            lines = len(file_content.split('\n'))
            st.metric("Lines", lines)

# ==================== SIDEBAR ====================
with st.sidebar:
    st.markdown("<h1 style='text-align: center;'>🚀 Control Panel</h1>", unsafe_allow_html=True)
    
    # Model Selection with icons
    st.subheader("🤖 Model Selection")
    models = {
        "🧠 Gemma 3 4B": "gemma3:4b",
        "⚡ Gemma 3 4B Q4": "gemma3:4b-instruct-q4_K_M",
        "🦙 Llama 2 7B": "llama2:7b",
        "🔍 DeepSeek R1": "deepseek-r1:7b",
        "🎭 Mixtral 8x7B": "mixtral:8x7b",
        "💎 CodeLlama": "codellama:7b"
    }
    
    selected_model = st.selectbox(
        "Choose Model",
        list(models.keys()),
        format_func=lambda x: f"{x}",
        help="Select AI model for conversation"
    )
    st.session_state.model = models[selected_model]
    
    # Advanced Settings in expander
    with st.expander("⚙️ Advanced Settings", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            st.session_state.temperature = st.slider(
                "Temperature", 0.0, 1.0, 0.7, 0.1,
                help="Creativity level"
            )
        with col2:
            context_length = st.select_slider(
                "Context Length",
                options=[2048, 4096, 8192, 16384],
                value=4096,
                help="Memory size"
            )
        
        streaming_speeds = {
            "🐢 Slow": "slow",
            "🚶 Medium": "medium", 
            "🚀 Fast": "fast",
            "⚡ Instant": "instant"
        }
        st.session_state.streaming_speed = st.selectbox(
            "Streaming Speed",
            list(streaming_speeds.keys())
        )
    
    # Feature Toggles
    st.subheader("🎛️ Feature Toggles")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.session_state.rag_enabled = st.toggle("📚 RAG", help="Document search")
    with col2:
        st.session_state.voice_enabled = st.toggle("🎤 Voice", disabled=not AUDIO_ENABLED)
    with col3:
        st.session_state.image_analysis = st.toggle("🖼️ Vision", disabled=not IMAGE_ENABLED)
    
    col4, col5, col6 = st.columns(3)
    with col4:
        st.session_state.tts_enabled = st.toggle("🔊 TTS", help="Text-to-speech")
    with col5:
        st.session_state.auto_refresh = st.toggle("🔄 Auto-refresh")
    with col6:
        dark_mode = st.toggle("🌙 Dark", True)
    
    # Plugin Selection
    st.subheader("🧩 Plugins")
    plugins = {
        "📧 Email Writer": "email",
        "📊 Data Analyzer": "data",
        "💻 Code Assistant": "code",
        "🎨 Creative Writer": "creative",
        "🔍 Research Assistant": "research",
        "📝 Translator": "translate"
    }
    
    selected_plugins = st.multiselect(
        "Enable Plugins",
        list(plugins.keys()),
        default=["💻 Code Assistant", "📧 Email Writer"]
    )
    st.session_state.selected_plugins = [plugins[p] for p in selected_plugins]
    
    # Workspace Management
    st.subheader("🗂️ Workspaces")
    workspace_col1, workspace_col2 = st.columns(2)
    with workspace_col1:
        new_workspace = st.text_input("New workspace")
        if st.button("➕") and new_workspace:
            st.session_state.workspaces[new_workspace] = []
            st.session_state.current_workspace = new_workspace
            st.rerun()
    
    with workspace_col2:
        workspace_list = list(st.session_state.workspaces.keys())
        selected_workspace = st.selectbox(
            "Current",
            workspace_list,
            index=workspace_list.index(st.session_state.current_workspace)
        )
        st.session_state.current_workspace = selected_workspace
    
    # Quick Actions
    st.subheader("⚡ Quick Actions")
    
    quick_col1, quick_col2 = st.columns(2)
    with quick_col1:
        if st.button("🧹 Clear Chat", use_container_width=True):
            st.session_state.messages = []
            st.success("Chat cleared!")
            time.sleep(1)
            st.rerun()
        
        if st.button("💾 Export", use_container_width=True):
            export_chat()
    
    with quick_col2:
        if st.button("📋 Templates", use_container_width=True):
            show_templates()
        
        if st.button("🔄 Restart", use_container_width=True):
            restart_session()
    
    # Status Panel
    st.subheader("📊 Status")
    
    status_col1, status_col2 = st.columns(2)
    with status_col1:
        try:
            models = ollama.list()
            st.metric("Models", len(models['models']))
        except:
            st.metric("Models", "❌")
    
    with status_col2:
        st.metric("Messages", len(st.session_state.messages))
    
    # Connection Status
    try:
        ollama.list()
        st.success("✅ Ollama Connected")
    except:
        st.error("❌ Ollama Not Connected")
    
    # Auto-refresh if enabled
    if st.session_state.auto_refresh:
        st_autorefresh(interval=5000, key="autorefresh")

with st.sidebar:
    if st.button("\U0001f504 Reset All Data", type="secondary"):
        st.session_state.chat_history = []
        st.session_state.messages = []
        st.session_state.favorites = []
        st.success("All data reset!")
        time.sleep(1)
        st.rerun()

# ==================== MAIN CONTENT ====================
# Tab System
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "💬 Chat", 
    "🎤 Voice", 
    "🖼️ Vision", 
    "👥 Collaborate", 
    "📊 Analytics",
    "⚙️ Settings"
])

# ==================== TAB 1: CHAT INTERFACE ====================
with tab1:
    st.markdown("<h1 class='main-header'>💬 Gemma 3 4B Chat</h1>", unsafe_allow_html=True)
    
    # Top Bar with Stats
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Model", selected_model.split()[0])
    with col2:
        st.metric("Context", f"{context_length:,}")
    with col3:
        st.metric("Temperature", st.session_state.temperature)
    with col4:
        st.metric("Plugins", len(selected_plugins))
    
    # Chat Container
    chat_container = st.container(height=500, border=True)
    
    with chat_container:
        # Display Messages
        for i, message in enumerate(st.session_state.messages):
            if message["role"] == "user":
                st.markdown(f"""
                <div style='display: flex; justify-content: flex-end; margin: 10px;'>
                    <div class='user-message'>
                        <strong>👤 You:</strong><br>
                        {message["content"]}
                    </div>
                </div>
                """, unsafe_allow_html=True)
            else:
                # Check if message has thinking process
                if "thinking" in message:
                    with st.expander("🧠 Thinking Process", expanded=False):
                        st.code(message["thinking"])
                
                st.markdown(f"""
                <div style='display: flex; justify-content: flex-start; margin: 10px;'>
                    <div class='ai-message'>
                        <strong>🤖 {selected_model.split()[0]}:</strong><br>
                        {message["content"]}
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                # Action buttons for each message
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    if st.button("📋 Copy", key=f"copy_{i}"):
                        st.write("Copied!")
                with col2:
                    if st.button("⭐ Save", key=f"save_{i}"):
                        st.session_state.favorites.append(message)
                        st.success("Saved to favorites!")
                with col3:
                    if st.button("🔄 Regenerate", key=f"regenerate_{i}"):
                        regenerate_message(i)
                with col4:
                    if st.button("📤 Share", key=f"share_{i}"):
                        share_message(message)
    
    # Input Area
    input_container = st.container()
    
    with input_container:
        col1, col2 = st.columns([6, 1])
        
        with col1:
            prompt = st.chat_input("Type your message here...", key="chat_input")
    
            if prompt:
                # Tampilkan pesan user langsung
                with st.chat_message("user"):
                    st.write(prompt)
                
                # Simpan ke messages
                st.session_state.messages.append({"role": "user", "content": prompt})
                
                # Dapatkan response dari AI
                with st.chat_message("assistant"):
                    with st.spinner("Thinking..."):
                        try:
                            # Siapkan messages untuk Ollama
                            messages_for_ollama = []
                            for msg in st.session_state.messages[-10:]:
                                messages_for_ollama.append({
                                    "role": msg["role"], 
                                    "content": msg["content"]
                                })
                            
                            # Dapatkan response
                            response = ollama.chat(
                                model=st.session_state.model,
                                messages=messages_for_ollama,
                                options={"temperature": st.session_state.temperature}
                            )
                            
                            # Tampilkan response
                            ai_response = response['message']['content']
                            st.write(ai_response)
                            
                            # Simpan ke messages
                            st.session_state.messages.append({
                                "role": "assistant", 
                                "content": ai_response
                            })
                            
                            # Simpan ke chat_history dengan format yang benar
                            st.session_state.chat_history.append({
                                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                "model": st.session_state.model,
                                "user": prompt,  # PERUBAHAN: dari 'user_prompt' ke 'user'
                                "assistant": ai_response,
                                "response_length": len(ai_response),
                                "response_time": "N/A"  # Placeholder
                            })
                            
                        except Exception as e:
                            error_msg = f"Error: {str(e)}"
                            st.error(error_msg)
                            st.session_state.messages.append({
                                "role": "assistant", 
                                "content": f"Sorry, I encountered an error: {str(e)}"
                            })
        
        with col2:
            # Quick action buttons
            quick_col1, quick_col2, quick_col3 = st.columns(3)
            
            with quick_col1:
                if st.button("\U0001f3a4", help="Voice input", disabled=not AUDIO_ENABLED, use_container_width=True):
                    # Toggle voice input
                    st.session_state.show_voice_input = not st.session_state.get('show_voice_input', False)
            
            with quick_col2:
                if st.button("\U0001f4ce", help="Attach file", use_container_width=True):
                    # Toggle file upload
                    st.session_state.show_file_upload = not st.session_state.get('show_file_upload', False)
            
            with quick_col3:
                if st.button("\U0001f3b2", help="Random prompt", use_container_width=True):
                    random_prompt = get_random_prompt()
                    st.session_state.messages.append({"role": "user", "content": random_prompt})
                    send_message(random_prompt)

        # Tampilkan voice input jika diaktifkan
        if st.session_state.get('show_voice_input', False):
            voice_input()

        # Tampilkan file upload jika diaktifkan
        if st.session_state.get('show_file_upload', False):
            file_upload()
    
    # Plugin Panel
    if st.session_state.selected_plugins:
        st.subheader("🧩 Active Plugins")
        
        plugin_cols = st.columns(min(3, len(st.session_state.selected_plugins)))
        for idx, plugin in enumerate(st.session_state.selected_plugins):
            with plugin_cols[idx % 3]:
                with st.expander(f"{get_plugin_icon(plugin)} {plugin.title()}", expanded=False):
                    if plugin == "email":
                        email_writer_ui()
                    elif plugin == "code":
                        code_assistant_ui()
                    elif plugin == "data":
                        data_analyzer_ui()
                    elif plugin == "creative":
                        creative_writer_ui()

# ==================== TAB 2: VOICE INTERFACE ====================
with tab2:
    if not AUDIO_ENABLED:
        st.warning("Audio features require additional packages. Install with: `pip install streamlit-audio-recorder speechrecognition pydub`")
    else:
        st.markdown("<h1 class='main-header'>🎤 Voice Interface</h1>", unsafe_allow_html=True)
        
        # Initialize Audio Processor
        if 'audio_processor' not in st.session_state:
            st.session_state.audio_processor = AudioProcessor()
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("🎙️ Voice Input")
            
            # Voice Recording
            recording_col1, recording_col2, recording_col3 = st.columns(3)
            
            with recording_col1:
                if st.button("🎤 Start Recording", type="primary", use_container_width=True):
                    with st.spinner("Recording..."):
                        audio_file = st.session_state.audio_processor.start_recording()
                        if audio_file:
                            st.success(f"Recording saved!")
            
            with recording_col2:
                if st.button("⏹️ Stop", use_container_width=True):
                    st.session_state.audio_processor.stop_recording()
            
            with recording_col3:
                if st.button("🎵 Playback", use_container_width=True):
                    audio_file = st.session_state.audio_processor.play_last_recording()
                    if audio_file:
                        st.audio(audio_file)
            
            # Audio File Upload
            uploaded_audio = st.file_uploader("Or upload audio file", type=['wav', 'mp3', 'm4a'])
            if uploaded_audio:
                # Save uploaded file
                audio_bytes = uploaded_audio.read()
                with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp:
                    tmp.write(audio_bytes)
                    tmp_path = tmp.name
                
                # Transcribe
                if st.button("Transcribe Uploaded Audio"):
                    with st.spinner("Transcribing..."):
                        text = st.session_state.audio_processor.transcribe_audio(tmp_path)
                        st.text_area("Transcription", text, height=100)
                        
                        # Auto-send to chat
                        if text and st.checkbox("Send to chat"):
                            st.session_state.messages.append({"role": "user", "content": f"[Voice] {text}"})
                            send_message(text)
            
            # Voice Settings
            with st.expander("⚙️ Voice Settings"):
                voice_speed = st.slider("Playback Speed", 0.5, 2.0, 1.0, 0.1)
                voice_volume = st.slider("Volume", 0.0, 1.0, 0.7, 0.1)
                language = st.selectbox("Language", ["en-US", "id-ID", "es-ES", "fr-FR"])
        
        with col2:
            st.subheader("🔊 Text-to-Speech")
            
            if st.session_state.tts_enabled:
                tts_text = st.text_area("Text to speak", height=150)
                
                tts_col1, tts_col2 = st.columns(2)
                with tts_col1:
                    if st.button("🗣️ Speak", use_container_width=True) and tts_text:
                        with st.spinner("Generating speech..."):
                            audio_path = st.session_state.audio_processor.text_to_speech(tts_text)
                            if audio_path:
                                st.audio(audio_path)
                
                with tts_col2:
                    voice_options = ["Male", "Female", "Neutral"]
                    selected_voice = st.selectbox("Voice", voice_options)
                
                # Save audio
                if st.button("💾 Save Audio", use_container_width=True) and tts_text:
                    audio_path = st.session_state.audio_processor.text_to_speech(tts_text, save=True)
                    if audio_path:
                        with open(audio_path, "rb") as f:
                            audio_bytes = f.read()
                        st.download_button(
                            "Download",
                            audio_bytes,
                            file_name=f"tts_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp3",
                            mime="audio/mp3"
                        )
            
            # Voice History
            st.subheader("📁 Voice History")
            if hasattr(st.session_state.audio_processor, 'history'):
                for i, item in enumerate(st.session_state.audio_processor.history[-5:]):
                    with st.expander(f"Recording {i+1} - {item['timestamp']}"):
                        st.write(item['text'][:100] + "...")
                        if st.button("▶️ Play", key=f"play_{i}"):
                            st.audio(item['path'])

# ==================== TAB 3: VISION INTERFACE ====================
with tab3:
    if not IMAGE_ENABLED:
        st.warning("""
        \U0001f5bc\ufe0f Vision features require OpenCV.
        
        **To enable:** 
        ```bash
        pip install opencv-python-headless==4.8.1.78 pillow pytesseract
        ```
        
        **Note:** OpenCV may have compatibility issues with NumPy 1.x.
        Consider using basic image features without OpenCV.
        """)
        
        # Basic image features without OpenCV
        st.subheader("\U0001f4f8 Basic Image Upload")
        uploaded_image = st.file_uploader("Upload Image", type=['png', 'jpg', 'jpeg'])
        
        if uploaded_image:
            from PIL import Image
            image = Image.open(uploaded_image)
            st.image(image, caption="Uploaded Image", use_column_width=True)
            
            # Basic info without OpenCV
            st.write(f"**Size:** {image.size[0]}x{image.size[1]}")
            st.write(f"**Format:** {image.format}")
            st.write(f"**Mode:** {image.mode}")
        
        st.stop()
    else:
        st.markdown("<h1 class='main-header'>🖼️ Vision & Image Analysis</h1>", unsafe_allow_html=True)
        
        # Initialize Image Analyzer
        if 'image_analyzer' not in st.session_state:
            st.session_state.image_analyzer = ImageAnalyzer()
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("📸 Image Upload & Analysis")
            
            # Image upload
            uploaded_image = st.file_uploader(
                "Upload Image",
                type=['png', 'jpg', 'jpeg', 'bmp', 'gif'],
                help="Upload image for analysis"
            )
            
            if uploaded_image:
                # Display image
                image = Image.open(uploaded_image)
                st.image(image, caption="Uploaded Image", use_column_width=True)
                
                # Get image info
                img_info = st.session_state.image_analyzer.get_image_info(image)
                
                info_col1, info_col2, info_col3 = st.columns(3)
                with info_col1:
                    st.metric("Size", f"{img_info['size'][0]}x{img_info['size'][1]}")
                with info_col2:
                    st.metric("Format", img_info['format'])
                with info_col3:
                    st.metric("Mode", img_info['mode'])
                
                # Analysis options
                analysis_options = st.multiselect(
                    "Select Analyses",
                    [
                        "🔤 Extract Text (OCR)",
                        "🎨 Describe Image",
                        "🏷️ Detect Objects",
                        "😀 Face Detection",
                        "🌈 Color Analysis",
                        "📏 Measure Objects"
                    ],
                    default=["🔤 Extract Text (OCR)", "🎨 Describe Image"]
                )
                
                # Perform analyses
                if st.button("🔍 Analyze Image", type="primary", use_container_width=True):
                    results = {}
                    
                    with st.spinner("Analyzing image..."):
                        # OCR if selected
                        if "🔤 Extract Text (OCR)" in analysis_options:
                            text = st.session_state.image_analyzer.extract_text(image)
                            results["OCR"] = text
                        
                        # Image description using LLM
                        if "🎨 Describe Image" in analysis_options:
                            description = describe_image_with_ai(image)
                            results["Description"] = description
                        
                        # Object detection
                        if "🏷️ Detect Objects" in analysis_options:
                            objects = st.session_state.image_analyzer.detect_objects(image)
                            results["Objects"] = objects
                        
                        # Color analysis
                        if "🌈 Color Analysis" in analysis_options:
                            colors = st.session_state.image_analyzer.analyze_colors(image)
                            results["Colors"] = colors
                    
                    # Display results
                    for analysis_type, result in results.items():
                        with st.expander(f"{analysis_type} Results", expanded=True):
                            if analysis_type == "Colors":
                                # Display color palette
                                colors = result
                                col_count = min(5, len(colors))
                                color_cols = st.columns(col_count)
                                for idx, (color, percentage) in enumerate(colors[:col_count]):
                                    with color_cols[idx]:
                                        st.color_picker(f"{percentage}%", color, disabled=True)
                                        st.caption(f"{color} ({percentage}%)")
                            else:
                                st.write(result)
            
            # Webcam capture
            st.subheader("📷 Webcam Capture")
            if st.button("Open Camera", use_container_width=True):
                camera_input = st.camera_input("Take a picture")
                if camera_input:
                    image = Image.open(camera_input)
                    st.image(image, caption="Captured Image", use_column_width=True)
        
        with col2:
            st.subheader("🛠️ Image Tools")
            
            # Image editing tools
            tool = st.selectbox(
                "Select Tool",
                ["Resize", "Crop", "Rotate", "Filter", "Convert Format"]
            )
            
            if uploaded_image:
                image = Image.open(uploaded_image)
                
                if tool == "Resize":
                    new_width = st.slider("Width", 100, 2000, image.width)
                    new_height = st.slider("Height", 100, 2000, image.height)
                    
                    if st.button("Resize"):
                        resized = st.session_state.image_analyzer.resize_image(image, new_width, new_height)
                        st.image(resized, caption=f"Resized to {new_width}x{new_height}")
                        
                        # Download resized image
                        buf = io.BytesIO()
                        resized.save(buf, format='PNG')
                        st.download_button(
                            "Download Resized",
                            buf.getvalue(),
                            file_name="resized_image.png",
                            mime="image/png"
                        )
                
                elif tool == "Filter":
                    filter_type = st.selectbox("Filter Type", ["Grayscale", "Blur", "Sharpen", "Edge Enhance"])
                    if st.button("Apply Filter"):
                        filtered = st.session_state.image_analyzer.apply_filter(image, filter_type)
                        st.image(filtered, caption=f"{filter_type} Filter")
            
            # Batch processing
            st.subheader("🔄 Batch Processing")
            batch_files = st.file_uploader(
                "Upload multiple images",
                type=['png', 'jpg', 'jpeg'],
                accept_multiple_files=True
            )
            
            if batch_files and st.button("Process Batch"):
                progress_bar = st.progress(0)
                results = []
                
                for i, file in enumerate(batch_files):
                    img = Image.open(file)
                    # Process each image
                    text = st.session_state.image_analyzer.extract_text(img)
                    results.append({
                        "filename": file.name,
                        "text": text[:100] + "..." if len(text) > 100 else text
                    })
                    progress_bar.progress((i + 1) / len(batch_files))
                
                # Display results
                df = pd.DataFrame(results)
                st.dataframe(df, use_container_width=True)
                
                # Download results
                csv = df.to_csv(index=False)
                st.download_button(
                    "Download Results",
                    csv,
                    file_name="batch_processing_results.csv",
                    mime="text/csv"
                )

# ==================== TAB 4: COLLABORATION INTERFACE ====================
with tab4:
    if not COLLAB_ENABLED:
        st.warning("Collaboration features require additional packages.")
    else:
        st.markdown("<h1 class='main-header'>👥 Real-time Collaboration</h1>", unsafe_allow_html=True)
        
        # Initialize Collaboration Session
        if st.session_state.collab_session is None:
            st.session_state.collab_session = CollaborationSession()
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            # Session Management
            st.subheader("🚀 Collaboration Session")
            
            session_col1, session_col2 = st.columns(2)
            with session_col1:
                session_name = st.text_input("Session Name", "Team Brainstorming")
                session_type = st.selectbox("Session Type", ["Brainstorming", "Coding", "Writing", "Research"])
            
            with session_col2:
                max_participants = st.number_input("Max Participants", 2, 50, 10)
                session_duration = st.select_slider("Duration", ["30min", "1hr", "2hr", "4hr", "Unlimited"])
            
            # Create/Join Session
            col1, col2, col3 = st.columns(3)
            with col1:
                if st.button("🆕 Create Session", type="primary", use_container_width=True):
                    session_id = st.session_state.collab_session.create_session(
                        name=session_name,
                        session_type=session_type,
                        max_users=max_participants
                    )
                    st.success(f"Session created! ID: {session_id}")
            
            with col2:
                join_id = st.text_input("Session ID to join")
                if st.button("🔗 Join Session", use_container_width=True) and join_id:
                    if st.session_state.collab_session.join_session(join_id):
                        st.success("Joined session!")
            
            with col3:
                if st.button("📋 Copy Invite Link", use_container_width=True):
                    invite_link = st.session_state.collab_session.get_invite_link()
                    st.code(invite_link)
            
            # Active Session Display
            if (st.session_state.collab_session is not None and 
                st.session_state.collab_session.active_session):
                st.subheader("💬 Collaborative Chat")
                
                # Display collaborative messages
                messages = st.session_state.collab_session.get_messages()
                
                for msg in messages[-20:]:  # Last 20 messages
                    with st.chat_message(msg["user"]):
                        st.write(f"**{msg['user']}:** {msg['message']}")
                        st.caption(f"{msg['timestamp']} • {msg.get('type', 'message')}")
                
                # Collaborative input
                collab_input = st.text_input("Type your message...", key="collab_input")
                col1, col2, col3 = st.columns([2, 1, 1])
                
                with col1:
                    if st.button("Send", key="collab_send") and collab_input:
                        st.session_state.collab_session.send_message(collab_input)
                        st.rerun()
                
                with col2:
                    if st.button("💡 Idea", key="collab_idea"):
                        idea = generate_collaborative_idea()
                        st.session_state.collab_session.send_message(f"💡 Idea: {idea}")
                        st.rerun()
                
                with col3:
                    if st.button("🤝 Vote", key="collab_vote"):
                        show_voting_interface()
                
                # Shared Whiteboard
                st.subheader("🖊️ Shared Whiteboard")
                
                whiteboard_col1, whiteboard_col2 = st.columns([3, 1])
                with whiteboard_col1:
                    # Simple whiteboard simulation
                    whiteboard_text = st.text_area(
                        "Collaborative Whiteboard",
                        height=200,
                        placeholder="Type here... everyone can see and edit in real-time!"
                    )
                    
                    if whiteboard_text != st.session_state.collab_session.whiteboard:
                        st.session_state.collab_session.update_whiteboard(whiteboard_text)
                
                with whiteboard_col2:
                    st.write("**Active Users:**")
                    users = st.session_state.collab_session.get_active_users()
                    for user in users:
                        st.write(f"👤 {user}")
                    
                    st.write("**Tools:**")
                    if st.button("🔄 Sync"):
                        st.rerun()
                    
                    if st.button("💾 Save"):
                        save_collaboration()
            
            # Task Management
            st.subheader("✅ Collaborative Tasks")
            
            if st.session_state.collab_session.active_session:
                # Add task
                task_col1, task_col2 = st.columns([3, 1])
                with task_col1:
                    new_task = st.text_input("New Task")
                with task_col2:
                    if st.button("Add Task") and new_task:
                        st.session_state.collab_session.add_task(new_task)
                
                # Display tasks
                tasks = st.session_state.collab_session.get_tasks()
                for task in tasks:
                    col1, col2, col3 = st.columns([6, 2, 1])
                    with col1:
                        st.checkbox(task["description"], value=task["completed"], key=f"task_{task['id']}")
                    with col2:
                        st.selectbox("Assignee", ["Unassigned"] + users, key=f"assignee_{task['id']}")
                    with col3:
                        if st.button("🗑️", key=f"delete_{task['id']}"):
                            st.session_state.collab_session.remove_task(task["id"])
        
        with col2:
            st.subheader("👥 Participants")
            
            if st.session_state.collab_session.active_session:
                participants = st.session_state.collab_session.get_participants()
                
                for participant in participants:
                    with st.expander(f"👤 {participant['name']}", expanded=False):
                        st.write(f"**Role:** {participant.get('role', 'Member')}")
                        st.write(f"**Joined:** {participant['joined']}")
                        st.write(f"**Messages:** {participant.get('message_count', 0)}")
                        
                        if st.button("Message", key=f"msg_{participant['id']}"):
                            private_message(participant['id'])
                
                # Participant stats
                st.metric("Active", len([p for p in participants if p.get('active')]))
                st.metric("Total Messages", sum(p.get('message_count', 0) for p in participants))
            
            st.subheader("📊 Session Analytics")
            
            if st.session_state.collab_session.active_session:
                # Generate some stats
                fig = go.Figure(data=[
                    go.Bar(
                        x=['Messages', 'Tasks', 'Ideas', 'Votes'],
                        y=[45, 12, 8, 15],
                        marker_color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
                    )
                ])
                fig.update_layout(title="Session Activity", height=300)
                st.plotly_chart(fig, use_container_width=True)
                
                # Export options
                if st.button("📤 Export Session Data"):
                    export_data = st.session_state.collab_session.export_data()
                    st.download_button(
                        "Download JSON",
                        json.dumps(export_data, indent=2),
                        file_name=f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                        mime="application/json"
                    )

# ==================== TAB 5: ANALYTICS ====================
with tab5:
    st.markdown("<h1 class='main-header'>\U0001f4ca Analytics & Insights</h1>", unsafe_allow_html=True)
    
    # Metrics Dashboard
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Chats", len(st.session_state.chat_history))
    
    with col2:
        if st.session_state.chat_history:
            # Hitung avg response length
            total_length = sum([msg.get('response_length', 0) for msg in st.session_state.chat_history])
            avg_length = total_length / len(st.session_state.chat_history) if len(st.session_state.chat_history) > 0 else 0
            st.metric("Avg Response Length", f"{avg_length:.0f} chars")
        else:
            st.metric("Avg Response Length", "0 chars")
    
    with col3:
        if st.session_state.chat_history:
            total_tokens = sum([msg.get('response_length', 0) for msg in st.session_state.chat_history])
            st.metric("Total Chars", f"{total_tokens:,}")
        else:
            st.metric("Total Chars", "0")
    
    with col4:
        favorite_count = len(st.session_state.favorites)
        st.metric("Favorites", favorite_count)
    
    # Charts
    chart_col1, chart_col2 = st.columns(2)
    
    with chart_col1:
        st.subheader("📈 Usage Over Time")
        
        # Create sample time series data
        dates = pd.date_range(start='2024-01-01', periods=30, freq='D')
        usage = np.random.randint(10, 100, size=30)
        
        fig = px.line(
            x=dates, 
            y=usage,
            title="Daily Chat Usage",
            labels={'x': 'Date', 'y': 'Messages'}
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with chart_col2:
        st.subheader("🧩 Plugin Usage")
        
        plugin_data = {
            'Plugin': ['Code', 'Email', 'Research', 'Creative', 'Translate'],
            'Usage': [45, 32, 28, 19, 12]
        }
        df = pd.DataFrame(plugin_data)
        
        fig = px.pie(
            df, 
            values='Usage', 
            names='Plugin',
            title="Plugin Usage Distribution",
            hole=0.3
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Detailed Analytics
    st.subheader("🔍 Detailed Analysis")
    
    analysis_tab1, analysis_tab2, analysis_tab3 = st.tabs(["Chat History", "Performance", "Cost Analysis"])
    
    with analysis_tab1:
        if st.session_state.chat_history:
            # Buat DataFrame
            df_history = pd.DataFrame(st.session_state.chat_history)
            
            # Cek kolom yang tersedia
            available_columns = df_history.columns.tolist()
            
            # Pilih kolom yang ada
            display_columns = []
            for col in ['timestamp', 'model', 'user', 'assistant', 'response_length', 'response_time']:
                if col in available_columns:
                    display_columns.append(col)
            
            if display_columns:
                st.dataframe(
                    df_history[display_columns],
                    use_container_width=True
                )
            else:
                st.info("No chat history data available")
            
            # Export options
            csv = df_history.to_csv(index=False)
            st.download_button(
                "\U0001f4e5 Export Full History",
                csv,
                file_name="full_chat_history.csv",
                mime="text/csv"
            )
    
    with analysis_tab2:
        # Performance metrics
        metrics = {
            'Metric': ['Avg Response Length', 'Max Response Time', 'Min Response Time', 'Success Rate'],
            'Value': ['245 chars', '8.2s', '1.1s', '98.5%']
        }
        st.table(pd.DataFrame(metrics))
        
        # Model comparison
        st.subheader("🤖 Model Performance Comparison")
        
        model_data = {
            'Model': ['Gemma 3 4B', 'Llama 2 7B', 'DeepSeek R1', 'Mixtral'],
            'Speed (tok/s)': [25, 15, 8, 5],
            'Accuracy (%)': [85, 82, 88, 90],
            'RAM Usage (GB)': [6, 12, 14, 32]
        }
        
        df_models = pd.DataFrame(model_data)
        st.dataframe(df_models, use_container_width=True)
    
    with analysis_tab3:
        # Cost estimation
        st.info("💡 Cost estimation based on equivalent cloud API pricing")
        
        col1, col2 = st.columns(2)
        
        with col1:
            input_tokens = st.number_input("Input tokens per request", 100, 10000, 500)
            output_tokens = st.number_input("Output tokens per request", 100, 10000, 500)
            requests_per_day = st.number_input("Requests per day", 10, 10000, 100)
        
        with col2:
            input_cost = st.number_input("Cost per input token ($)", 0.000001, 0.001, 0.000002, format="%.6f")
            output_cost = st.number_input("Cost per output token ($)", 0.000001, 0.001, 0.000002, format="%.6f")
        
        # Calculate
        daily_cost = (input_tokens * input_cost + output_tokens * output_cost) * requests_per_day
        monthly_cost = daily_cost * 30
        
        st.metric("Estimated Daily Cost", f"${daily_cost:.4f}")
        st.metric("Estimated Monthly Cost", f"${monthly_cost:.2f}")
        
        # Savings compared to cloud
        cloud_monthly = 20  # Example cloud subscription
        savings = cloud_monthly - monthly_cost
        st.metric("Savings vs Cloud", f"${savings:.2f}", delta=f"{savings/cloud_monthly*100:.1f}%")


# ==================== TAB 6: SETTINGS ====================
with tab6:
    st.markdown("<h1 class='main-header'>⚙️ Advanced Settings</h1>", unsafe_allow_html=True)
    
    settings_tab1, settings_tab2, settings_tab3, settings_tab4 = st.tabs([
        "General", "API", "Appearance", "Advanced"
    ])
    
    with settings_tab1:
        st.subheader("General Settings")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Auto-save settings
            auto_save = st.toggle("Auto-save conversations", True)
            save_interval = st.select_slider(
                "Save interval",
                options=["Every message", "Every 5 messages", "Every 10 messages", "Manual only"]
            )
            
            # Notifications
            notify_new = st.toggle("Notify on new messages", True)
            notify_sound = st.toggle("Play notification sound", True)
            
        with col2:
            # Privacy
            store_history = st.toggle("Store chat history", True)
            if not store_history:
                st.warning("Chat history will not be saved after session ends")
            
            # Data retention
            retention = st.select_slider(
                "Data retention period",
                options=["1 day", "1 week", "1 month", "3 months", "Forever"]
            )
            
            # Clear data button
            if st.button("🗑️ Clear All Data", type="secondary"):
                with st.expander("⚠️ Confirmation", expanded=True):
                    st.warning("This will delete:")
                    st.write("- All chat history")
                    st.write("- All uploaded files")
                    st.write("- All preferences")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        if st.button("Cancel"):
                            st.info("Operation cancelled")
                    with col2:
                        if st.button("Delete Everything", type="primary"):
                            # List of keys to keep
                            keep_keys = ['_last_run', '_widget_state']
                            
                            # Delete all other keys
                            for key in [k for k in st.session_state.keys() if k not in keep_keys]:
                                del st.session_state[key]
                            
                            st.success("All data cleared! Refreshing...")
                            time.sleep(2)
                            st.rerun()
    
    with settings_tab2:
        st.subheader("API & Integration Settings")
        
        # Ollama settings
        st.write("**Ollama Configuration**")
        
        ollama_host = st.text_input(
            "Ollama Host",
            value="http://localhost:11434",
            help="URL of your Ollama server"
        )
        
        ollama_timeout = st.number_input(
            "Request Timeout (seconds)",
            min_value=5,
            max_value=300,
            value=30
        )
        
        # External APIs
        st.write("**External API Keys**")
        
        api_keys = {
            "OpenAI": st.text_input("OpenAI API Key", type="password"),
            "Google AI": st.text_input("Google AI API Key", type="password"),
            "Anthropic": st.text_input("Anthropic API Key", type="password"),
            "Hugging Face": st.text_input("Hugging Face Token", type="password")
        }
        
        # Test connection
        if st.button("Test Connections"):
            test_api_connections()
    
    with settings_tab3:
        st.subheader("Appearance & Theme")
        
        # Theme selection
        theme = st.selectbox(
            "Theme",
            ["Dark", "Light", "Blue", "Green", "Purple", "Custom"]
        )
        
        if theme == "Custom":
            col1, col2, col3 = st.columns(3)
            with col1:
                primary_color = st.color_picker("Primary Color", "#FF4B4B")
            with col2:
                background_color = st.color_picker("Background", "#0E1117")
            with col3:
                text_color = st.color_picker("Text Color", "#FAFAFA")
        
        # Layout options
        st.write("**Layout Options**")
        
        layout = st.radio(
            "Layout Style",
            ["Compact", "Comfortable", "Spacious"],
            horizontal=True
        )
        
        font_size = st.slider("Font Size", 12, 24, 16)
        chat_bubble_style = st.selectbox(
            "Chat Bubble Style",
            ["Modern", "Classic", "Minimal", "Bubbles"]
        )
        
        # Preview - PERBAIKI DI SINI
        if st.button("Preview Theme"):
            preview_theme(theme)  # Fungsi ini sudah didefinisikan
    
    with settings_tab4:
        st.subheader("Advanced Configuration")
        
        # Model parameters
        st.write("**Model Parameters**")
        
        col1, col2 = st.columns(2)
        
        with col1:
            top_p = st.slider("Top-p", 0.0, 1.0, 0.9, 0.01)
            frequency_penalty = st.slider("Frequency Penalty", -2.0, 2.0, 0.0, 0.1)
        
        with col2:
            presence_penalty = st.slider("Presence Penalty", -2.0, 2.0, 0.0, 0.1)
            repeat_penalty = st.slider("Repeat Penalty", 1.0, 2.0, 1.1, 0.1)
        
        # System parameters
        st.write("**System Parameters**")
        
        max_threads = st.slider("Max Threads", 1, 16, 8)
        batch_size = st.slider("Batch Size", 1, 64, 32)
        cache_size = st.select_slider(
            "Cache Size",
            options=["256MB", "512MB", "1GB", "2GB", "4GB"]
        )
        
        # Experimental features
        st.write("**Experimental Features**")
        
        experimental_features = {
            "Multi-model routing": st.toggle("Route to best model", False),
            "Auto-model switching": st.toggle("Switch models based on task", False),
            "Predictive loading": st.toggle("Pre-load likely models", False),
            "Federated learning": st.toggle("Learn from usage patterns", False)
        }
        
        # Danger zone
        with st.expander("⚠️ Danger Zone", expanded=False):
            st.warning("These settings can break the application")
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Reset to Defaults", type="secondary"):
                    reset_to_defaults()
            
            with col2:
                if st.button("Factory Reset", type="primary"):
                    factory_reset()

# ==================== HELPER FUNCTIONS ====================
#def send_message(prompt: str):
#    """Send message to Ollama"""
#    if not prompt.strip():
#        return
    
    # Add user message to display
#    st.session_state.messages.append({"role": "user", "content": prompt})
    
#    try:
        # Prepare messages for Ollama
#        messages_for_ollama = st.session_state.messages[-10:]  # Last 10 messages
        
        # Get response
#        response = ollama.chat(
#            model=st.session_state.model,
#            messages=messages_for_ollama,
#            options={"temperature": st.session_state.temperature}
#        )
        
        # Add assistant response
#        st.session_state.messages.append({
#            "role": "assistant", 
#            "content": response['message']['content']
#        })
        
        # Add to chat history
#        st.session_state.chat_history.append({
#            "timestamp": datetime.now(),
#            "user": prompt,
#            "assistant": response['message']['content'],
#            "model": st.session_state.model
#        })
        
        # Rerun to update display
#        st.rerun()
        
#    except Exception as e:
#        st.error(f"Error: {str(e)}")
#        st.session_state.messages.append({
#            "role": "assistant", 
#            "content": f"Sorry, I encountered an error: {str(e)}"
#        })



def get_plugin_icon(plugin: str) -> str:
    """Get icon for plugin"""
    icons = {
        "email": "📧",
        "code": "💻",
        "data": "📊",
        "creative": "🎨",
        "research": "🔍",
        "translate": "🌐"
    }
    return icons.get(plugin, "🧩")

def email_writer_ui():
    """Email writing plugin UI"""
    recipient = st.text_input("Recipient")
    subject = st.text_input("Subject")
    tone = st.selectbox("Tone", ["Formal", "Casual", "Friendly", "Professional"])
    key_points = st.text_area("Key Points")
    
    if st.button("Generate Email"):
        prompt = f"""
        Write a {tone.lower()} email:
        To: {recipient}
        Subject: {subject}
        Key points: {key_points}
        """
        
        with st.spinner("Writing email..."):
            response = ollama.chat(
                model=st.session_state.model,
                messages=[{"role": "user", "content": prompt}]
            )
            st.text_area("Generated Email", response['message']['content'], height=200)

def code_assistant_ui():
    """Code assistant plugin UI"""
    language = st.selectbox("Language", ["Python", "JavaScript", "Java", "C++", "Go", "Rust"])
    task = st.selectbox("Task", ["Write function", "Debug", "Optimize", "Explain", "Convert"])
    code_input = st.text_area("Your code", height=150)
    
    if st.button("Get Help"):
        prompt = f"""
        As a {language} expert, {task}:
        
        {code_input}
        
        Provide clear, concise help.
        """
        
        with st.spinner("Analyzing code..."):
            response = ollama.chat(
                model=st.session_state.model,
                messages=[{"role": "user", "content": prompt}]
            )
            st.code(response['message']['content'], language=language.lower())

def describe_image_with_ai(image):
    """Use AI to describe image"""
    # Convert image to base64
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    
    # Use vision model if available
    try:
        response = ollama.chat(
            model="llava:7b",  # Vision model
            messages=[{
                "role": "user",
                "content": "Describe this image in detail",
                "images": [img_str]
            }]
        )
        return response['message']['content']
    except:
        return "Image description not available. Install a vision model like LLaVA."

def generate_collaborative_idea():
    """Generate idea for collaboration"""
    ideas = [
        "Let's brainstorm solutions for climate change",
        "How about we create a new programming language?",
        "Let's design the future of education",
        "Brainstorm ideas for a sustainable city",
        "Create a plan for Mars colonization"
    ]
    return np.random.choice(ideas)

def private_message(user_id: str):
    """Send private message in collaboration"""
    message = st.text_input(f"Private message to {user_id}")
    if st.button("Send"):
        st.success(f"Sent to {user_id}")

def save_collaboration():
    """Save collaboration session"""
    data = st.session_state.collab_session.export_data()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    with open(f"collaboration_{timestamp}.json", "w") as f:
        json.dump(data, f, indent=2)
    
    st.success("Collaboration saved!")

def export_chat():
    """Export chat history"""
    if st.session_state.chat_history:
        # Buat DataFrame
        df = pd.DataFrame(st.session_state.chat_history)
        
        # Pilih format yang tersedia
        col1, col2 = st.columns(2)
        
        with col1:
            csv = df.to_csv(index=False)
            st.download_button(
                "\U0001f4e5 CSV",
                csv,
                file_name=f"chat_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
        
        with col2:
            json_str = df.to_json(indent=2, orient="records")
            st.download_button(
                "\U0001f4e5 JSON",
                json_str,
                file_name=f"chat_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )
    else:
        st.info("No chat history to export")

def play_notification():
    """Play notification sound"""
    # Simple beep sound using HTML5 audio
    audio_html = """
    <audio autoplay>
        <source src="https://assets.mixkit.co/sfx/preview/mixkit-correct-answer-tone-2870.mp3" type="audio/mpeg">
    </audio>
    """
    st.components.v1.html(audio_html, height=0)

# ==================== MAIN EXECUTION ====================
if __name__ == "__main__":
    # Auto-start Ollama check
    try:
        ollama.list()
    except:
        st.sidebar.error("⚠️ Ollama not running. Start with: `ollama serve`")
    
    # Welcome message on first run
    if len(st.session_state.messages) == 0:
        welcome_msg = "Hello! I'm your Gemma 3 4B assistant. How can I help you today?"
        st.session_state.messages.append({
            "role": "assistant",
            "content": welcome_msg,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })