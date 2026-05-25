import os
import json
import subprocess
from pathlib import Path

# ── Paths ────────────────────────────────────────────────────────────────
BASE_DIR  = Path(__file__).resolve().parent.parent
VIDEO_DIR = BASE_DIR / "videos"
AUDIO_DIR = BASE_DIR / "audio"
DATA_FILE = BASE_DIR / "data" / "videos.json"
AUDIO_DIR.mkdir(exist_ok=True)

# ── Check ffmpeg is installed ─────────────────────────────────────────────
def check_ffmpeg():
    try:
        result = subprocess.run(
            ["ffmpeg", "-version"],
            capture_output=True, text=True
        )
        if result.returncode == 0:
            print("  ✅ ffmpeg is available")
            return True
    except FileNotFoundError:
        pass
    print("  ❌ ffmpeg not found. Please install it and add to PATH.")
    print("     Windows: winget install ffmpeg")
    print("     Or download from: https://ffmpeg.org/download.html")
    return False


# ── Extract audio from a single video ────────────────────────────────────
def extract_audio(video: dict) -> dict:
    video_id   = video["id"]
    title      = video["title"]
    video_path = VIDEO_DIR / f"{video_id}.mp4"
    audio_path = AUDIO_DIR / f"{video_id}.wav"

    print(f"\n{'─'*55}")
    print(f"  Video : {video_id} — {title}")
    print(f"{'─'*55}")

    # Check video file exists
    if not video_path.exists():
        print(f"  ❌ Video file not found: {video_path}")
        print(f"     Make sure {video_id}.mp4 is in the videos/ folder")
        return {**video, "status": "failed", "audio_file": None}

    # Skip if already extracted
    if audio_path.exists() and audio_path.stat().st_size > 1000:
        print(f"  ✅ Audio already extracted, skipping.")
        return {**video, "status": "skipped", "audio_file": str(audio_path)}

    print(f"  🎵 Extracting audio from {video_id}.mp4 ...")

    try:
        # ffmpeg command:
        # -i          input file
        # -vn         no video
        # -acodec     audio codec (pcm_s16le = standard WAV)
        # -ar         sample rate 16000hz (optimal for Whisper)
        # -ac         mono channel (reduces file size, fine for speech)
        # -y          overwrite output without asking
        command = [
            "ffmpeg",
            "-i", str(video_path),
            "-vn",
            "-acodec", "pcm_s16le",
            "-ar", "16000",
            "-ac", "1",
            "-y",
            str(audio_path)
        ]

        result = subprocess.run(
            command,
            capture_output=True,
            text=True
        )

        if result.returncode != 0:
            print(f"  ❌ ffmpeg error:\n{result.stderr[-500:]}")
            return {**video, "status": "failed", "audio_file": None}

        # Verify output file
        if not audio_path.exists() or audio_path.stat().st_size < 1000:
            print(f"  ❌ Audio file missing or empty after extraction")
            return {**video, "status": "failed", "audio_file": None}

        size_mb = audio_path.stat().st_size / (1024 * 1024)
        print(f"  ✅ Audio extracted successfully ({size_mb:.1f} MB)")
        print(f"     Saved to: audio/{video_id}.wav")
        return {**video, "status": "success", "audio_file": str(audio_path)}

    except Exception as e:
        print(f"  ❌ Unexpected error: {e}")
        return {**video, "status": "failed", "audio_file": None}


# ── Extract audio from all videos ────────────────────────────────────────
def extract_all_audio() -> list:

    # Check ffmpeg first
    if not check_ffmpeg():
        return []

    # Load video list
    with open(DATA_FILE, "r") as f:
        videos = json.load(f)

    print(f"\n🎵 Starting audio extraction for {len(videos)} videos...\n")

    results = []
    for video in videos:
        result = extract_audio(video)
        results.append(result)

    # ── Summary ──────────────────────────────────────────────────────────
    success = [r for r in results if r["status"] == "success"]
    skipped = [r for r in results if r["status"] == "skipped"]
    failed  = [r for r in results if r["status"] == "failed"]

    print(f"\n{'═'*55}")
    print(f"  AUDIO EXTRACTION SUMMARY")
    print(f"{'═'*55}")
    print(f"  ✅ Extracted : {len(success)}")
    print(f"  ⏭️  Skipped   : {len(skipped)} (already existed)")
    print(f"  ❌ Failed    : {len(failed)}")

    if failed:
        print(f"\n  ⚠️  Failed videos:")
        for v in failed:
            print(f"     - {v['id']} : {v['title']}")
            print(f"       Make sure {v['id']}.mp4 exists in videos/ folder")

    print(f"{'═'*55}\n")

    # Save log
    log_path = BASE_DIR / "data" / "audio_log.json"
    with open(log_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  📄 Log saved to: data/audio_log.json\n")

    return results


if __name__ == "__main__":
    extract_all_audio()