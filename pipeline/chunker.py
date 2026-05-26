import json
from pathlib import Path

# Paths
BASE_DIR         = Path(__file__).resolve().parent.parent
TRANSCRIPTS_DIR  = BASE_DIR / "transcripts"
OCR_DIR          = BASE_DIR / "ocr_output"
VISION_DIR       = BASE_DIR / "visual_analysis"
CHUNKS_DIR       = BASE_DIR / "chunks"
DATA_FILE        = BASE_DIR / "data" / "videos.json"
CHUNKS_DIR.mkdir(exist_ok=True)

# Config 
MAX_CHUNK_WORDS = 300   # Max words per chunk
OVERLAP_WORDS   = 50    # Overlap between chunks to preserve context


# Load transcript segments 
def load_transcript(video_id: str) -> list:
    path = TRANSCRIPTS_DIR / f"{video_id}_segments.json"
    if not path.exists():
        print(f"   Transcript not found: {path}")
        return []
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# Load OCR results
def load_ocr(video_id: str) -> list:
    path = OCR_DIR / f"{video_id}_ocr.json"
    if not path.exists():
        print(f"     ⚠️  OCR not found: {path}")
        return []
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return [d for d in data if d.get("has_text")]


# Load vision results 
def load_vision(video_id: str) -> list:
    path = VISION_DIR / f"{video_id}_vision.json"
    if not path.exists():
        print(f"   Vision not found: {path}")
        return []
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return [d for d in data if d.get("has_visual_content")]


# Find OCR and vision data near a timestamp 
def get_nearby_visual(data: list, start_sec: float,
                      end_sec: float, window: int = 15) -> list:
    """
    Return visual data whose timestamp falls within
    the transcript segment range plus a small window.
    """
    matches = []
    for item in data:
        ts = item.get("timestamp", 0)
        if (start_sec - window) <= ts <= (end_sec + window):
            matches.append(item)
    return matches


# Build chunks from transcript + OCR + vision 
def build_chunks(video: dict) -> list:
    video_id = video["id"]
    title    = video["title"]
    url      = video["url"]

    segments = load_transcript(video_id)
    ocr_data = load_ocr(video_id)
    vis_data = load_vision(video_id)

    if not segments:
        print(f"   No transcript segments found for {video_id}")
        return []

    chunks       = []
    chunk_id     = 0
    current_segs = []
    current_words = 0

    def flush_chunk(segs: list):
        nonlocal chunk_id
        if not segs:
            return

        #  Transcript text
        transcript_text = " ".join(s["text"] for s in segs).strip()
        start_sec       = segs[0]["start_sec"]
        end_sec         = segs[-1]["end_sec"]
        start_fmt       = segs[0]["start_fmt"]
        end_fmt         = segs[-1]["end_fmt"]

        #Nearby OCR text
        nearby_ocr = get_nearby_visual(ocr_data, start_sec, end_sec)
        ocr_texts  = []
        for item in nearby_ocr:
            text = item.get("ocr_text", "").strip()
            ts   = item.get("timestamp_fmt", "")
            if text:
                ocr_texts.append(f"[Screen text at {ts}]: {text}")
        ocr_combined = "\n".join(ocr_texts)

        #Nearby vision descriptions
        nearby_vis = get_nearby_visual(vis_data, start_sec, end_sec)
        vis_texts  = []
        for item in nearby_vis:
            desc = item.get("visual_description", "").strip()
            ts   = item.get("timestamp_fmt", "")
            if desc:
                vis_texts.append(f"[Visual at {ts}]: {desc}")
        vis_combined = "\n".join(vis_texts)

        #Build full chunk text
        chunk_parts = [f"[Transcript {start_fmt} → {end_fmt}]: {transcript_text}"]
        if ocr_combined:
            chunk_parts.append(ocr_combined)
        if vis_combined:
            chunk_parts.append(vis_combined)

        full_text = "\n\n".join(chunk_parts)

        chunks.append({
            "chunk_id"          : f"{video_id}_chunk_{chunk_id:03d}",
            "video_id"          : video_id,
            "title"             : title,
            "url"               : url,
            "start_sec"         : start_sec,
            "end_sec"           : end_sec,
            "start_fmt"         : start_fmt,
            "end_fmt"           : end_fmt,
            "timestamp"         : f"{start_fmt} → {end_fmt}",
            "transcript_text"   : transcript_text,
            "ocr_text"          : ocr_combined,
            "visual_description": vis_combined,
            "full_text"         : full_text,
            "word_count"        : len(full_text.split())
        })
        chunk_id += 1

    # Group segments into chunks
    for seg in segments:
        word_count = len(seg["text"].split())

        # If adding this segment exceeds limit, flush current chunk
        if current_words + word_count > MAX_CHUNK_WORDS and current_segs:
            flush_chunk(current_segs)

            # Keep last few segments for overlap
            overlap_words = 0
            overlap_segs  = []
            for s in reversed(current_segs):
                overlap_words += len(s["text"].split())
                overlap_segs.insert(0, s)
                if overlap_words >= OVERLAP_WORDS:
                    break

            current_segs  = overlap_segs
            current_words = overlap_words

        current_segs.append(seg)
        current_words += word_count

    # Flush remaining segments
    flush_chunk(current_segs)

    return chunks


# Process chunks for a single video
def process_video_chunks(video: dict) -> dict:
    video_id = video["id"]
    title    = video["title"]

    chunks_path = CHUNKS_DIR / f"{video_id}_chunks.json"

    print(f"\n{'─'*55}")
    print(f"  Video : {video_id} — {title}")
    print(f"{'─'*55}")

    # Skip if already done
    if chunks_path.exists():
        with open(chunks_path) as f:
            existing = json.load(f)
        print(f"   Already chunked ({len(existing)} chunks), skipping.")
        return {**video, "status": "skipped",
                "chunks_file": str(chunks_path),
                "chunk_count": len(existing)}

    chunks = build_chunks(video)

    if not chunks:
        return {**video, "status": "failed", "chunks_file": None}

    # Save chunks
    with open(chunks_path, "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=2, ensure_ascii=False)

    # Save readable version
    readable_path = CHUNKS_DIR / f"{video_id}_chunks_readable.txt"
    with open(readable_path, "w", encoding="utf-8") as f:
        f.write(f"VIDEO: {title}\n")
        f.write(f"URL  : {video['url']}\n")
        f.write(f"Total chunks: {len(chunks)}\n")
        f.write(f"{'═'*55}\n\n")
        for chunk in chunks:
            f.write(f"CHUNK: {chunk['chunk_id']}\n")
            f.write(f"Time : {chunk['timestamp']}\n")
            f.write(f"Words: {chunk['word_count']}\n")
            f.write(f"{'─'*40}\n")
            f.write(f"{chunk['full_text']}\n")
            f.write(f"{'═'*55}\n\n")

    # Print preview
    avg_words = sum(c["word_count"] for c in chunks) // len(chunks)
    print(f"   Chunking complete")
    print(f"     Chunks created : {len(chunks)}")
    print(f"     Avg words/chunk: {avg_words}")
    print(f"     Saved to       : chunks/{video_id}_chunks.json")

    # Print first chunk as sample
    if chunks:
        print(f"\n   Sample chunk preview:")
        print(f"     {chunks[0]['full_text'][:200]}...")

    return {
        **video,
        "status"     : "success",
        "chunks_file": str(chunks_path),
        "chunk_count": len(chunks)
    }


#  Process all videos 
def process_all_chunks() -> list:
    with open(DATA_FILE, "r") as f:
        videos = json.load(f)

    print(f"\nStarting chunking for {len(videos)} videos...\n")

    results = []
    for video in videos:
        result = process_video_chunks(video)
        results.append(result)

    success = [r for r in results if r["status"] == "success"]
    skipped = [r for r in results if r["status"] == "skipped"]
    failed  = [r for r in results if r["status"] == "failed"]

    total_chunks = sum(r.get("chunk_count", 0) for r in success + skipped)

    print(f"\n{'═'*55}")
    print(f"  CHUNKING SUMMARY")
    print(f"{'═'*55}")
    print(f"  Processed   : {len(success)} videos")
    print(f"   Skipped     : {len(skipped)} (already existed)")
    print(f"   Failed      : {len(failed)}")
    print(f"   Total chunks: {total_chunks}")
    print(f"{'═'*55}\n")

    log_path = BASE_DIR / "data" / "chunks_log.json"
    with open(log_path, "w") as f:
        json.dump(
            [{k: v for k, v in r.items()} for r in results],
            f, indent=2
        )
    print(f"   Log saved to: data/chunks_log.json\n")

    return results


#  Run directly
if __name__ == "__main__":
    process_all_chunks()