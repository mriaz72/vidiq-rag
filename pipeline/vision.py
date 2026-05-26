import json
import os
import base64
import time
from pathlib import Path
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

# Groq setup 
client       = Groq(api_key=os.getenv("GROQ_API_KEY_2"))
VISION_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"

# Paths
BASE_DIR         = Path(__file__).resolve().parent.parent
KEYFRAMES_DIR    = BASE_DIR / "keyframes"
VISION_DIR       = BASE_DIR / "visual_analysis"
DATA_FILE        = BASE_DIR / "data" / "videos.json"
VISION_DIR.mkdir(exist_ok=True)


# Encode image to base64 
def encode_image(filepath: str) -> str:
    with open(filepath, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


# Analyze a single frame
def analyze_frame(frame_info: dict) -> dict:
    filepath = frame_info.get("filepath")

    if not filepath or not os.path.exists(filepath):
        return {**frame_info, "visual_description": "", "has_visual_content": False}

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
                                "text": """Analyze this video frame and describe what you observe.
Focus on:

1. SPEAKER: What is the speaker doing? Are they gesturing, pointing, 
   emphasizing something, or showing strong emotion?

2. ON SCREEN CONTENT: Are there any slides, documents, charts, forms, 
   websites, or screen recordings visible? If yes describe their content.

3. VISUAL EMPHASIS: Is anything being highlighted, underlined, or 
   pointed to that seems important?

4. OVERALL CONTEXT: What is the main topic or message being communicated 
   in this frame based on what you see?

Be specific and concise. Focus only on what is actually visible.
Do not guess or invent details that are not in the image."""
                            }
                        ]
                    }
                ],
                max_tokens=300
            )

            description = response.choices[0].message.content.strip()

            if not description or len(description) < 10:
                return {
                    **frame_info,
                    "visual_description"  : "",
                    "has_visual_content"  : False
                }

            return {
                **frame_info,
                "visual_description" : description,
                "has_visual_content" : True
            }

        except Exception as e:
            error_str = str(e)
            if "429" in error_str or "rate" in error_str.lower():
                wait_time = (attempt + 1) * 15
                print(f"      Rate limit. Waiting {wait_time}s... "
                      f"(retry {attempt+1}/{max_retries})")
                time.sleep(wait_time)
            else:
                print(f"      Vision failed for {filepath}: {e}")
                return {
                    **frame_info,
                    "visual_description" : "",
                    "has_visual_content" : False
                }

    print(f"    All retries exhausted for {filepath}")
    return {**frame_info, "visual_description": "", "has_visual_content": False}


# Process visual analysis for a single video 
def analyze_video(video: dict) -> dict:
    video_id = video["id"]
    title    = video["title"]

    keyframes_log = KEYFRAMES_DIR / f"{video_id}_keyframes.json"
    vision_path   = VISION_DIR / f"{video_id}_vision.json"

    print(f"\n{'─'*55}")
    print(f"  Video : {video_id} — {title}")
    print(f"{'─'*55}")

    if not keyframes_log.exists():
        print(f"   Keyframes log not found. Run keyframer.py first")
        return {**video, "status": "failed", "vision_file": None}

    # Skip if already done
    if vision_path.exists():
        with open(vision_path) as f:
            existing = json.load(f)
        print(f"   Already analyzed ({len(existing)} frames), skipping.")
        return {**video, "status": "skipped", "vision_file": str(vision_path)}

    with open(keyframes_log) as f:
        keyframes = json.load(f)

    print(f"   Analyzing {len(keyframes)} frames with Groq Vision...")

    vision_results      = []
    frames_with_content = 0

    for i, frame in enumerate(keyframes):
        result = analyze_frame(frame)
        vision_results.append(result)

        if result["has_visual_content"]:
            frames_with_content += 1
            preview = result["visual_description"][:80].replace("\n", " ")
            print(f"     [{i+1}/{len(keyframes)}]  "
                  f"{frame['timestamp_fmt']} — {preview}...")
        else:
            print(f"     [{i+1}/{len(keyframes)}] "
                  f"{frame['timestamp_fmt']} — no visual content")

        # Respect Groq rate limits
        time.sleep(3)

    # Save JSON
    with open(vision_path, "w", encoding="utf-8") as f:
        json.dump(vision_results, f, indent=2, ensure_ascii=False)

    # Save readable TXT
    readable_path = VISION_DIR / f"{video_id}_vision_readable.txt"
    with open(readable_path, "w", encoding="utf-8") as f:
        f.write(f"VIDEO: {title}\n")
        f.write(f"URL  : {video['url']}\n")
        f.write(f"{'─'*55}\n\n")
        for r in vision_results:
            if r["has_visual_content"]:
                f.write(f"[{r['timestamp_fmt']}] — {r['reason']}\n")
                f.write(f"{r['visual_description']}\n")
                f.write(f"{'─'*30}\n\n")

    print(f"\n   Visual analysis complete")
    print(f"     Total frames        : {len(keyframes)}")
    print(f"     Frames with content : {frames_with_content}")

    return {
        **video,
        "status"              : "success",
        "vision_file"         : str(vision_path),
        "total_frames"        : len(keyframes),
        "frames_with_content" : frames_with_content
    }


#  Analyze all videos
def analyze_all_videos() -> list:
    with open(DATA_FILE, "r") as f:
        videos = json.load(f)

    print(f"\n Starting visual analysis for {len(videos)} videos...\n")

    results = []
    for video in videos:
        result = analyze_video(video)
        results.append(result)

    success = [r for r in results if r["status"] == "success"]
    skipped = [r for r in results if r["status"] == "skipped"]
    failed  = [r for r in results if r["status"] == "failed"]

    total_frames  = sum(r.get("total_frames", 0) for r in success)
    total_content = sum(r.get("frames_with_content", 0) for r in success)

    print(f"\n{'═'*55}")
    print(f"  VISUAL ANALYSIS SUMMARY")
    print(f"{'═'*55}")
    print(f"  Processed        : {len(success)} videos")
    print(f"    Skipped          : {len(skipped)} (already existed)")
    print(f"  Failed           : {len(failed)}")
    print(f"  Total frames     : {total_frames}")
    print(f"   Frames analyzed  : {total_content}")
    print(f"{'═'*55}\n")

    log_path = BASE_DIR / "data" / "vision_log.json"
    with open(log_path, "w") as f:
        json.dump(
            [{k: v for k, v in r.items() if k != "frames"} for r in results],
            f, indent=2
        )
    print(f"  Log saved to: data/vision_log.json\n")

    return results


# Run directly 
if __name__ == "__main__":
    analyze_all_videos()