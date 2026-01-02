# 🚀 Gemma 3 4B Chat UI

A feature-rich Streamlit interface for Ollama with Gemma 3 4B model.

## ✨ Features

- 💬 Chat interface with multiple AI models
- 🎤 Voice input and TTS support
- 🖼️ Image analysis and processing  
- 👥 Real-time collaboration
- 📊 Analytics and insights
- ⚙️ Customizable settings

## 🛠️ Installation

### 1. Install Ollama
```bash
curl -fsSL https://ollama.com/install.sh | sh
```

### 2. Pull Gemma 3 4B model
```bash
ollama pull gemma3:4b
```

### 3. Clone repository
```bash
git clone https://github.com/yourusername/ollama-gemma3-ui.git
cd ollama-gemma3-ui
```

### 4. Install Python dependencies
```bash
pip install -r requirements.txt
```

### 5. Run the application
```bash
streamlit run app.py
```

## 📋 Requirements

  - Python 3.10+

  - Ollama running on port 11434

  - Gemma 3 4B model (or other compatible models)

## 🚀 Quick Start

 1. Start Ollama server:
    ```bash
    ollama serve
    ```

2. In another terminal:
   ```bash
   cd ollama-gemma3-ui
   streamlit run app.py
   ```

3. Open browser: http://localhost:8501

## 🗂️ Project Structure

 ```text
 ollama-gemma3-ui/
├── app.py              # Main application
├── requirements.txt    # Python dependencies
├── README.md          # Documentation
├── .gitignore         # Git ignore file
├── audio_handler.py   # Audio processing module
├── image_processor.py # Image analysis module
└── collaborative.py   # Collaboration module
```

## 📸 Screenshots

<img width="1895" height="932" alt="Screenshot_2026-01-02_14-41-30" src="https://github.com/user-attachments/assets/d835e616-cec4-438b-ac20-2dd1ade33510" />
<img width="1906" height="927" alt="Screenshot_2026-01-02_14-41-48" src="https://github.com/user-attachments/assets/733df281-d32f-47f1-b92a-7b3290cff9f0" />

## 🤝 Contributing

Pull requests are welcome! For major changes, please open an issue first.
