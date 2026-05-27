# VidIQ-RAG — Multimodal AI Video Consultant

A RAG-based AI consultant tool that processes videos end-to-end and answers
questions strictly from video content with full source citations.

## Demo

Run locally:
```bash
streamlit run app/streamlit_app.py
```

## Tech Stack

| Component | Tool |
|---|---|
| Video download | yt-dlp + Playwright |
| Audio extraction | ffmpeg |
| Transcription | Groq Whisper large-v3 |
| Keyframe extraction | OpenCV + PySceneDetect |
| OCR | Groq Vision (llama-4-scout) |
| Visual analysis | Groq Vision (llama-4-scout) |
| Embeddings | sentence-transformers all-MiniLM-L6-v2 |
| Vector store | ChromaDB (local) |
| Q&A answers | Groq LLaMA 3.3 70b |
| Demo UI | Streamlit |

## Project Structure
vidiq-rag/
├── pipeline/
│   ├── downloader.py     # Video download
│   ├── extractor.py      # Audio extraction
│   ├── transcriber.py    # Whisper transcription
│   ├── keyframer.py      # Keyframe extraction
│   ├── ocr.py            # OCR on frames
│   ├── vision.py         # Visual analysis
│   ├── chunker.py        # Chunk builder
│   ├── embedder.py       # Embeddings + ChromaDB
│   └── qa.py             # Retrieval + Q&A
├── app/
│   └── streamlit_app.py  # Demo UI
├── data/
│   └── videos.json       # Video URLs
├── transcripts/          # Whisper transcripts
├── keyframes/            # Extracted frames
├── ocr_output/           # OCR results
├── visual_analysis/      # Vision descriptions
├── chunks/               # RAG chunks
├── vectorstore/          # ChromaDB storage
└── outputs/              # Sample Q&A results
## Setup

### 1. Clone the repo
```bash
git clone https://github.com/YOUR_USERNAME/vidiq-rag.git
cd vidiq-rag
```

### 2. Create virtual environment
```bash
uv venv
# Windows:
.venv\Scripts\activate
# Mac/Linux:
source .venv/bin/activate
```

### 3. Install dependencies
```bash
uv add yt-dlp ffmpeg-python scenedetect pytesseract pillow chromadb \
google-genai python-dotenv groq opencv-python sentence-transformers \
streamlit playwright
```

### 4. Add API keys to `.env`
GROQ_API_KEY=your_groq_key_here
### 5. Add videos to `videos/` folder
Name them `video_01.mp4` through `video_10.mp4`

### 6. Run pipeline in order
```bash
python pipeline/extractor.py
python pipeline/transcriber.py
python pipeline/keyframer.py
python pipeline/ocr.py
python pipeline/vision.py
python pipeline/chunker.py
python pipeline/embedder.py
```

### 7. Run demo
```bash
streamlit run app/streamlit_app.py
```

## Pipeline Flow
Video files
↓
Audio extraction (ffmpeg)
↓
Transcription with timestamps (Groq Whisper)
↓
Keyframe extraction (OpenCV + PySceneDetect)
↓
OCR on frames (Groq Vision)
↓
Visual analysis (Groq Vision)
↓
Chunk builder — transcript + OCR + visual combined
↓
Embeddings + ChromaDB storage
↓
RAG retrieval + LLM answer generation
↓
Streamlit demo UI
## Anti-Hallucination

The system answers strictly from retrieved video chunks only.
The LLM is instructed to respond with:
"I don't know based on the provided video content."
when the answer is not found in the videos.

## Scaling Plan

| Scale | Approach |
|---|---|
| 10 videos | Local machine, current setup |
| 100 videos | Parallel workers, Qdrant Cloud |
| 1000 videos | Celery + Redis task queue, cloud storage |
| 14000 videos | Distributed pipeline, Prefect orchestration |

## Sample Q&A Results

See `outputs/sample_qa_readable.txt` for 10 answered questions
with source citations and 2 no-answer refusal tests.