import os
import json
import time
from pathlib import Path
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

# Paths
BASE_DIR        = Path(__file__).resolve().parent.parent
AUDIO_DIR       = BASE_DIR / "audio"
TRANSCRIPTS_DIR = BASE_DIR / "transcripts"
DATA_FILE       = BASE_DIR / "data" / "videos.json"
TRANSCRIPTS_DIR.mkdir(exist_ok=True)

# Groq client 
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# Transcribe a single audio file 
def transcribe_audio(video: dict) -> dict:
    video_id   = video["id"]
    title      = video["title"]
    url        = video["url"]
    audio_path = AUDIO_DIR / f"{video_id}.wav"

    # Output paths
    raw_json_path   = TRANSCRIPTS_DIR / f"{video_id}_raw.json"
    clean_txt_path  = TRANSCRIPTS_DIR / f"{video_id}_clean.txt"
    chunks_json_path = TRANSCRIPTS_DIR / f"{video_id}_segments.json"

    print(f"\n{'─'*55}")
    print(f"  Video : {video_id} — {title}")
    print(f"{'─'*55}")

    # Check audio file exists
    if not audio_path.exists():
        print(f"  ❌ Audio file not found: {audio_path}")
        print(f"     Run extractor.py first")
        return {**video, "status": "failed", "transcript_file": None}

    # Skip if already transcribed
    if raw_json_path.exists() and clean_txt_path.exists():
        print(f"  ✅ Already transcribed, skipping.")
        return {
            **video,
            "status": "skipped",
            "transcript_file": str(clean_txt_path),
            "segments_file": str(chunks_json_path)
        }

    print(f"  🎙️  Sending to Groq Whisper API...")

    try:
        # Call Groq Whisper API
        with open(audio_path, "rb") as audio_file:
            response = client.audio.transcriptions.create(
                file=(f"{video_id}.wav", audio_file),
                model="whisper-large-v3",
                response_format="verbose_json",  # gives us word timestamps
                language="en",
                temperature=0.0
            )

        #  Save raw response
        raw_data = {
            "video_id"   : video_id,
            "title"      : title,
            "url"        : url,
            "duration"   : getattr(response, "duration", None),
            "language"   : getattr(response, "language", "en"),
            "full_text"  : response.text,
            "segments"   : []
        }

        # Extract segments with timestamps

        if hasattr(response, "segments") and response.segments:
            for i, seg in enumerate(response.segments):
                # Handle both object and dict format
                if isinstance(seg, dict):
                    raw_data["segments"].append({
                        "id"    : seg.get("id", i),
                        "start" : round(seg.get("start", 0), 2),
                        "end"   : round(seg.get("end", 0), 2),
                        "text"  : seg.get("text", "").strip()
                    })
                else:
                    raw_data["segments"].append({
                        "id"    : getattr(seg, "id", i),
                        "start" : round(getattr(seg, "start", 0), 2),
                        "end"   : round(getattr(seg, "end", 0), 2),
                        "text"  : getattr(seg, "text", "").strip()
                    })

        with open(raw_json_path, "w", encoding="utf-8") as f:
            json.dump(raw_data, f, indent=2, ensure_ascii=False)

        #  Build clean segments with formatted timestamps
        segments = []
        for seg in raw_data["segments"]:
            start_fmt = format_timestamp(seg["start"])
            end_fmt   = format_timestamp(seg["end"])
            segments.append({
                "video_id"   : video_id,
                "title"      : title,
                "url"        : url,
                "start_sec"  : seg["start"],
                "end_sec"    : seg["end"],
                "start_fmt"  : start_fmt,
                "end_fmt"    : end_fmt,
                "timestamp"  : f"{start_fmt} → {end_fmt}",
                "text"       : seg["text"]
            })

        with open(chunks_json_path, "w", encoding="utf-8") as f:
            json.dump(segments, f, indent=2, ensure_ascii=False)

        #  Save clean readable transcript
        with open(clean_txt_path, "w", encoding="utf-8") as f:
            f.write(f"VIDEO: {title}\n")
            f.write(f"URL  : {url}\n")
            f.write(f"{'─'*55}\n\n")
            for seg in segments:
                f.write(f"[{seg['timestamp']}]\n")
                f.write(f"{seg['text']}\n\n")

        #  Print preview 
        word_count = len(response.text.split())
        seg_count  = len(segments)
        print(f"    Transcription complete")
        print(f"     Segments : {seg_count}")
        print(f"     Words    : {word_count}")
        print(f"     Preview  : {response.text[:100]}...")

        # Small delay to respect Groq rate limits
        time.sleep(1)

        return {
            **video,
            "status"         : "success",
            "transcript_file": str(clean_txt_path),
            "segments_file"  : str(chunks_json_path),
            "word_count"     : word_count,
            "segment_count"  : seg_count
        }

    except Exception as e:
        print(f"   Transcription failed: {e}")
        return {**video, "status": "failed", "transcript_file": None}


#  Format seconds to MM:SS 
def format_timestamp(seconds: float) -> str:
    seconds = int(seconds)
    minutes = seconds // 60
    secs    = seconds % 60
    return f"{minutes:02d}:{secs:02d}"


#  Transcribe all videos 
def transcribe_all() -> list:
    with open(DATA_FILE, "r") as f:
        videos = json.load(f)

    print(f"\n  Starting transcription for {len(videos)} videos...\n")

    results = []
    for video in videos:
        result = transcribe_audio(video)
        results.append(result)

    # Summary
    success = [r for r in results if r["status"] == "success"]
    skipped = [r for r in results if r["status"] == "skipped"]
    failed  = [r for r in results if r["status"] == "failed"]

    print(f"\n{'═'*55}")
    print(f"  TRANSCRIPTION SUMMARY")
    print(f"{'═'*55}")
    print(f"   Transcribed : {len(success)}")
    print(f"    Skipped     : {len(skipped)} (already existed)")
    print(f"   Failed      : {len(failed)}")

    if failed:
        print(f"\n  ⚠️  Failed videos:")
        for v in failed:
            print(f"     - {v['id']} : {v['title']}")

    total_words = sum(r.get("word_count", 0) for r in success)
    print(f"\n   Total words transcribed : {total_words}")
    print(f"{'═'*55}\n")

    # Save log
    log_path = BASE_DIR / "data" / "transcription_log.json"
    with open(log_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"   Log saved to: data/transcription_log.json\n")

    return results


# Run directly 
if __name__ == "__main__":
    transcribe_all()