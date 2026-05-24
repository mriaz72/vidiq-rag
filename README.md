# VidIQ-RAG

A multimodal RAG-based AI consultant tool that processes videos end-to-end —
audio, transcription, keyframes, OCR, and visual analysis — and answers
questions strictly from video content with source citations.

## Tech Stack

- **Transcription** — Groq Whisper API
- **Visual Analysis** — Gemini 1.5 Flash API
- **Embeddings** — sentence-transformers (all-MiniLM-L6-v2)
- **Q&A** — Gemini 1.5 Flash API
- **Vector Store** — ChromaDB (local)
- **OCR** — Tesseract
- **Keyframes** — OpenCV + PySceneDetect

## Project Structure
vidiq-rag/
├── pipeline/        # Core processing scripts
├── app/             # Streamlit demo UI
├── data/            # Video URLs and logs
├── transcripts/     # Whisper transcripts
├── keyframes/       # Extracted video frames
├── ocr_output/      # OCR results
├── visual_analysis/ # Gemini vision output
├── chunks/          # RAG chunks
├── vectorstore/     # ChromaDB storage
└── outputs/         # Final Q&A results
## Setup

1. Clone the repo
```bash
   git clone https://github.com/yourusername/vidiq-rag.git
   cd vidiq-rag
```

2. Create virtual environment
```bash
   uv venv
   source .venv/bin/activate  # Windows: .venv\Scripts\activate
```

3. Install dependencies
```bash
   uv add yt-dlp ffmpeg-python scenedetect pytesseract pillow chromadb \
   google-generativeai python-dotenv groq opencv-python sentence-transformers
```

4. Add API keys to `.env`
GEMINI_API_KEY=your_key_here
GROQ_API_KEY=your_key_here
5. Add your videos to `videos/` folder named `video_01.mp4` through `video_10.mp4`

6. Run the pipeline step by step
```bash
   python pipeline/extractor.py
   python pipeline/transcriber.py
   python pipeline/keyframer.py
   python pipeline/ocr.py
   python pipeline/vision.py
   python pipeline/chunker.py
   python pipeline/embedder.py
```

7. Launch the demo
```bash
   streamlit run app/streamlit_app.py
```

## Anti-Hallucination

The system answers strictly from retrieved video chunks only.
If the answer is not found it responds with:
"I don't know based on the provided video content."


