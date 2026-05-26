import json
import os
import base64
import time
from pathlib import Path
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

#Groq setup 
client = Groq(api_key=os.getenv("GROQ_API_KEY"))
VISION_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"

# Paths 
BASE_DIR      = Path(__file__).resolve().parent.parent
KEYFRAMES_DIR = BASE_DIR / "keyframes"
OCR_DIR       = BASE_DIR / "ocr_output"
DATA_FILE     = BASE_DIR / "data" / "videos.json"
OCR_DIR.mkdir(exist_ok=True)


# Encode image to base64 ─
def encode_image(filepath: str) -> str:
    with open(filepath, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


# Run OCR on a single frame ─
def run_ocr_on_frame(frame_info: dict) -> dict:
    filepath = frame_info.get("filepath")

    if not filepath or not os.path.exists(filepath):
        return {**frame_info, "ocr_text": "", "has_text": False}

    max_retries = 3
    for attempt in range(max_retries):
        try:
            image_data = encode_image(filepath)

            response = client.chat.completions.create(
                model=VISION_MODEL,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{image_data}"
                                }
                            },
                            {
                                "type": "text",
                                "text": """Extract all visible text from this video frame exactly as it appears.
Include:
- Titles and headings
- Bullet points and lists
- Any text overlays or captions
- Show name or branding

Return only the extracted text, line by line.
If no text is visible, return exactly: NO_TEXT"""
                            }
                        ]
                    }
                ]
            )

            extracted = response.choices[0].message.content.strip()

            if extracted == "NO_TEXT" or len(extracted) < 3:
                return {**frame_info, "ocr_text": "", "has_text": False}

            # Clean up any preamble like "The text in the image is:"
            lines = extracted.splitlines()
            cleaned_lines = []
            skip_phrases = [
                "the text in the image",
                "the visible text",
                "here is the text",
                "extracted text",
                "the image contains",
                "the image shows"
            ]
            for line in lines:
                if any(phrase in line.lower() for phrase in skip_phrases):
                    continue
                if line.strip():
                    cleaned_lines.append(line.strip())

            final_text = "\n".join(cleaned_lines).strip()

            if not final_text:
                return {**frame_info, "ocr_text": "", "has_text": False}

            return {
                **frame_info,
                "ocr_text": final_text,
                "has_text": True
            }

        except Exception as e:
            error_str = str(e)
            if "429" in error_str or "rate" in error_str.lower():
                wait_time = (attempt + 1) * 15
                print(f"     Rate limit. Waiting {wait_time}s... "
                      f"(retry {attempt+1}/{max_retries})")
                time.sleep(wait_time)
            else:
                print(f"     OCR failed for {filepath}: {e}")
                return {**frame_info, "ocr_text": "", "has_text": False}

    print(f"      All retries exhausted for {filepath}")
    return {**frame_info, "ocr_text": "", "has_text": False}


#  Process OCR for a single video 
def process_video_ocr(video: dict) -> dict:
    video_id = video["id"]
    title    = video["title"]

    keyframes_log   = KEYFRAMES_DIR / f"{video_id}_keyframes.json"
    ocr_output_path = OCR_DIR / f"{video_id}_ocr.json"

    print(f"\n{'─'*55}")
    print(f"  Video : {video_id} — {title}")
    print(f"{'─'*55}")

    if not keyframes_log.exists():
        print(f"   Keyframes log not found. Run keyframer.py first")
        return {**video, "status": "failed", "ocr_file": None}

    if ocr_output_path.exists():
        with open(ocr_output_path) as f:
            existing = json.load(f)
        frames_with_text = sum(1 for f in existing if f.get("has_text"))
        print(f"  Already processed ({len(existing)} frames, "
              f"{frames_with_text} with text), skipping.")
        return {**video, "status": "skipped", "ocr_file": str(ocr_output_path)}

    with open(keyframes_log) as f:
        keyframes = json.load(f)

    print(f"  Running Groq Vision OCR on {len(keyframes)} frames...")

    ocr_results      = []
    frames_with_text = 0

    for i, frame in enumerate(keyframes):
        result = run_ocr_on_frame(frame)
        ocr_results.append(result)

        if result["has_text"]:
            frames_with_text += 1
            preview = result["ocr_text"][:80].replace("\n", " ")
            print(f"     [{i+1}/{len(keyframes)}]  "
                  f"{frame['timestamp_fmt']} — {preview}...")
        else:
            print(f"     [{i+1}/{len(keyframes)}] "
                  f"{frame['timestamp_fmt']} — no text")

        # Respect Groq free tier rate limit
        time.sleep(3)

    # Save JSON
    with open(ocr_output_path, "w", encoding="utf-8") as f:
        json.dump(ocr_results, f, indent=2, ensure_ascii=False)

    # Save readable TXT
    readable_path = OCR_DIR / f"{video_id}_ocr_readable.txt"
    with open(readable_path, "w", encoding="utf-8") as f:
        f.write(f"VIDEO: {title}\n")
        f.write(f"URL  : {video['url']}\n")
        f.write(f"{'─'*55}\n\n")
        for r in ocr_results:
            if r["has_text"]:
                f.write(f"[{r['timestamp_fmt']}] — {r['reason']}\n")
                f.write(f"{r['ocr_text']}\n")
                f.write(f"{'─'*30}\n\n")

    print(f"\n  OCR complete")
    print(f"     Total frames    : {len(keyframes)}")
    print(f"     Frames with text: {frames_with_text}")

    return {
        **video,
        "status"          : "success",
        "ocr_file"        : str(ocr_output_path),
        "total_frames"    : len(keyframes),
        "frames_with_text": frames_with_text
    }


#  Process all videos 
def process_all_ocr() -> list:
    with open(DATA_FILE, "r") as f:
        videos = json.load(f)

    print(f"\n Starting Groq Vision OCR for {len(videos)} videos...\n")

    results = []
    for video in videos:
        result = process_video_ocr(video)
        results.append(result)

    success = [r for r in results if r["status"] == "success"]
    skipped = [r for r in results if r["status"] == "skipped"]
    failed  = [r for r in results if r["status"] == "failed"]

    total_frames = sum(r.get("total_frames", 0) for r in success)
    total_text   = sum(r.get("frames_with_text", 0) for r in success)

    print(f"\n{'═'*55}")
    print(f"  OCR SUMMARY")
    print(f"{'═'*55}")
    print(f"  Processed     : {len(success)} videos")
    print(f"  Skipped       : {len(skipped)} (already existed)")
    print(f"  Failed        : {len(failed)}")
    print(f"   Total frames  : {total_frames}")
    print(f"  Frames w/ text: {total_text}")
    print(f"{'═'*55}\n")

    log_path = BASE_DIR / "data" / "ocr_log.json"
    with open(log_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Log saved to: data/ocr_log.json\n")

    return results


# Run directly 
if __name__ == "__main__":
    process_all_ocr()