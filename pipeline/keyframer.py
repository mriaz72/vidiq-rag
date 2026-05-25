import cv2
import json
import os
from pathlib import Path
from scenedetect import open_video, SceneManager
from scenedetect.detectors import ContentDetector

# Paths
BASE_DIR      = Path(__file__).resolve().parent.parent
VIDEO_DIR     = BASE_DIR / "videos"
KEYFRAMES_DIR = BASE_DIR / "keyframes"
DATA_FILE     = BASE_DIR / "data" / "videos.json"
KEYFRAMES_DIR.mkdir(exist_ok=True)

#  Extract keyframes for a single video 
def extract_keyframes(video: dict) -> dict:
    video_id   = video["id"]
    title      = video["title"]
    video_path = VIDEO_DIR / f"{video_id}.mp4"

    # Output folder per video
    video_frames_dir = KEYFRAMES_DIR / video_id
    video_frames_dir.mkdir(exist_ok=True)

    # Output log file
    log_path = KEYFRAMES_DIR / f"{video_id}_keyframes.json"

    print(f"\n{'─'*55}")
    print(f"  Video : {video_id} — {title}")
    print(f"{'─'*55}")

    # Check video exists
    if not video_path.exists():
        print(f"   Video file not found: {video_path}")
        return {**video, "status": "failed", "keyframes": []}

    # Skip if already done
    if log_path.exists():
        with open(log_path) as f:
            existing = json.load(f)
        print(f"   Already extracted ({len(existing)} frames), skipping.")
        return {**video, "status": "skipped", "keyframes": existing}

    keyframes = []

    try:
        #  Step 1: Scene change detection 
        print(f"   Detecting scene changes...")
        scene_timestamps = detect_scenes(str(video_path))
        print(f"     Found {len(scene_timestamps)} scene changes")

        #  Step 2: Interval-based timestamps (every 30 seconds) 
        print(f"    Adding interval frames (every 30 seconds)...")
        cap = cv2.VideoCapture(str(video_path))
        fps          = cap.get(cv2.CAP_PROP_FPS)
        total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        duration     = total_frames / fps if fps > 0 else 0
        cap.release()

        interval_timestamps = list(range(0, int(duration), 30))

        # Step 3: Merge and deduplicate timestamps
        all_timestamps = sorted(set(scene_timestamps + interval_timestamps))
        # Remove timestamps too close together (within 3 seconds)
        filtered = [all_timestamps[0]] if all_timestamps else []
        for ts in all_timestamps[1:]:
            if ts - filtered[-1] >= 3:
                filtered.append(ts)

        print(f"    Extracting {len(filtered)} frames total...")

        #  Step 4: Extract and save each frame 
        cap = cv2.VideoCapture(str(video_path))

        for ts in filtered:
            # Set video position to timestamp
            cap.set(cv2.CAP_PROP_POS_MSEC, ts * 1000)
            ret, frame = cap.read()

            if not ret:
                continue

            # Format timestamp for filename
            ts_fmt    = format_timestamp(ts)
            filename  = f"{video_id}_{ts_fmt.replace(':', 'm')}s.png"
            frame_path = video_frames_dir / filename

            # Save frame
            cv2.imwrite(str(frame_path), frame)

            # Check if frame has meaningful content (not black/blank)
            if is_blank_frame(frame):
                os.remove(str(frame_path))
                continue

            keyframes.append({
                "video_id"   : video_id,
                "title"      : title,
                "url"        : video["url"],
                "timestamp"  : ts,
                "timestamp_fmt": ts_fmt,
                "filename"   : filename,
                "filepath"   : str(frame_path),
                "reason"     : "scene_change" if ts in scene_timestamps else "interval"
            })

        cap.release()

        #  Save log 
        with open(log_path, "w") as f:
            json.dump(keyframes, f, indent=2)

        print(f"   Extracted {len(keyframes)} keyframes")
        print(f"     Saved to: keyframes/{video_id}/")

        return {**video, "status": "success", "keyframes": keyframes}

    except Exception as e:
        print(f"   Error: {e}")
        return {**video, "status": "failed", "keyframes": []}


# Detect scene changes using PySceneDetect 
def detect_scenes(video_path: str) -> list:
    try:
        video        = open_video(video_path)
        scene_manager = SceneManager()
        scene_manager.add_detector(ContentDetector(threshold=30.0))
        scene_manager.detect_scenes(video)
        scene_list = scene_manager.get_scene_list()

        # Extract start timestamp in seconds for each scene
        timestamps = []
        for scene in scene_list:
            start_sec = scene[0].get_seconds()
            timestamps.append(int(start_sec))

        return timestamps

    except Exception as e:
        print(f"       Scene detection failed: {e}, using interval only")
        return []


# Check if frame is blank or black 
def is_blank_frame(frame) -> bool:
    # Convert to grayscale and check mean brightness
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    mean_brightness = gray.mean()
    return mean_brightness < 10  # Nearly black frame


# Format seconds to MM:SS 
def format_timestamp(seconds: int) -> str:
    minutes = seconds // 60
    secs    = seconds % 60
    return f"{minutes:02d}:{secs:02d}"


# Extract keyframes from all videos 
def extract_all_keyframes() -> list:
    with open(DATA_FILE, "r") as f:
        videos = json.load(f)

    print(f"\n  Starting keyframe extraction for {len(videos)} videos...\n")

    results = []
    for video in videos:
        result = extract_keyframes(video)
        results.append(result)

    #  Summary
    success = [r for r in results if r["status"] == "success"]
    skipped = [r for r in results if r["status"] == "skipped"]
    failed  = [r for r in results if r["status"] == "failed"]

    total_frames = sum(len(r.get("keyframes", [])) for r in success)

    print(f"\n{'═'*55}")
    print(f"  KEYFRAME EXTRACTION SUMMARY")
    print(f"{'═'*55}")
    print(f"   Processed  : {len(success)} videos")
    print(f"    Skipped    : {len(skipped)} (already existed)")
    print(f"   Failed     : {len(failed)}")
    print(f"    Total frames: {total_frames}")

    if failed:
        print(f"\n    Failed videos:")
        for v in failed:
            print(f"     - {v['id']} : {v['title']}")

    print(f"{'═'*55}\n")

    # Save log
    log_path = BASE_DIR / "data" / "keyframes_log.json"
    with open(log_path, "w") as f:
        json.dump(
            [{k: v for k, v in r.items() if k != "keyframes"} for r in results],
            f, indent=2
        )
    print(f"   Log saved to: data/keyframes_log.json\n")

    return results


# Run directly 
if __name__ == "__main__":
    extract_all_keyframes()