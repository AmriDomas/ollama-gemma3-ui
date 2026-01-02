import uuid
from datetime import datetime
from typing import List, Dict
import json

class CollaborationSession:
    def __init__(self):
        self.sessions = {}
        self.active_session = None
    
    def create_session(self, name: str, session_type: str, max_users: int = 10):
        """Create new collaboration session"""
        session_id = str(uuid.uuid4())[:8]
        
        self.sessions[session_id] = {
            'id': session_id,
            'name': name,
            'type': session_type,
            'max_users': max_users,
            'created': datetime.now(),
            'users': [],
            'messages': [],
            'whiteboard': "",
            'tasks': []
        }
        
        self.active_session = session_id
        return session_id
    
    def join_session(self, session_id: str, username: str = None):
        """Join existing session"""
        if session_id in self.sessions:
            if username is None:
                username = f"User_{len(self.sessions[session_id]['users']) + 1}"
            
            user = {
                'id': str(uuid.uuid4())[:8],
                'name': username,
                'joined': datetime.now(),
                'active': True
            }
            
            self.sessions[session_id]['users'].append(user)
            self.active_session = session_id
            return True
        return False
    
    def send_message(self, message: str, user: str = "Anonymous"):
        """Send message to active session"""
        if self.active_session:
            msg = {
                'id': str(uuid.uuid4())[:8],
                'user': user,
                'message': message,
                'timestamp': datetime.now(),
                'type': 'message'
            }
            
            self.sessions[self.active_session]['messages'].append(msg)
            return True
        return False
    
    def get_messages(self):
        """Get messages from active session"""
        if self.active_session:
            return self.sessions[self.active_session]['messages']
        return []

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

    
    def get_active_users(self):
        """Get active users in session"""
        if self.active_session:
            return [u['name'] for u in self.sessions[self.active_session]['users']]
        return []
    
    def update_whiteboard(self, content: str):
        """Update whiteboard content"""
        if self.active_session:
            self.sessions[self.active_session]['whiteboard'] = content
            return True
        return False
    
    def add_task(self, description: str):
        """Add task to session"""
        if self.active_session:
            task = {
                'id': str(uuid.uuid4())[:8],
                'description': description,
                'created': datetime.now(),
                'completed': False,
                'assignee': None
            }
            
            self.sessions[self.active_session]['tasks'].append(task)
            return True
        return False
    
    def get_invite_link(self):
        """Generate invite link"""
        if self.active_session:
            return f"https://your-app.com/collab/{self.active_session}"
        return ""
    
    def export_data(self):
        """Export session data"""
        if self.active_session:
            return self.sessions[self.active_session]
        return {}

    def send_message(prompt: str):
        """Send message to Ollama"""
        if not prompt.strip():
            return
        
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        try:
            # Prepare messages for Ollama
            messages_for_ollama = st.session_state.messages[-10:]  # Last 10 messages
            
            # Get response
            response = ollama.chat(
                model=st.session_state.model,
                messages=messages_for_ollama,
                options={"temperature": st.session_state.temperature}
            )
            
            # Add assistant response
            st.session_state.messages.append({
                "role": "assistant", 
                "content": response['message']['content']
            })
            
            # Add to chat history
            st.session_state.chat_history.append({
                "timestamp": datetime.now(),
                "user": prompt,
                "assistant": response['message']['content'],
                "model": st.session_state.model
            })
            
        except Exception as e:
            st.error(f"Error: {str(e)}")
            st.session_state.messages.append({
                "role": "assistant", 
                "content": f"Sorry, I encountered an error: {str(e)}"
            })

    def export_chat():
        """Export chat history"""
        if st.session_state.chat_history:
            df = pd.DataFrame(st.session_state.chat_history)
            csv = df.to_csv(index=False)
            
            st.download_button(
                "📥 Download Chat History",
                csv,
                file_name=f"chat_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
        else:
            st.info("No chat history to export")

    # ==================== INITIALIZE SESSION STATE ====================

    def init_session_state():
        """Initialize all session state variables"""
        if "messages" not in st.session_state:
            st.session_state.messages = []
        
        if "model" not in st.session_state:
            st.session_state.model = "gemma3:4b-instruct"
        
        if "temperature" not in st.session_state:
            st.session_state.temperature = 0.7
        
        if "selected_plugins" not in st.session_state:
            st.session_state.selected_plugins = []
        
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []
