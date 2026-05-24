import yt_dlp
import json
import os
import subprocess
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Paths
BASE_DIR   = Path(__file__).resolve().parent.parent
DATA_FILE  = BASE_DIR / "data" / "videos.json"
VIDEO_DIR  = BASE_DIR / "videos"
VIDEO_DIR.mkdir(exist_ok=True)

#  yt-dlp options 
def get_ydl_opts(output_path: str) -> dict:
    return {
        "format": "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best",
        "outtmpl": output_path,
        "merge_output_format": "mp4",
        "quiet": False,
        "no_warnings": False,
        "extractor_args": {
            "rumble": {"formats": ["mp4"]}
        },
        "http_headers": {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            )
        },
        "retries": 3,
        "fragment_retries": 3,
        "socket_timeout": 30,
    }


# Primary download using yt-dlp 
def download_with_ytdlp(url: str, output_path: str) -> bool:
    try:
        print(f"    [yt-dlp] Attempting download...")
        with yt_dlp.YoutubeDL(get_ydl_opts(output_path)) as ydl:
            ydl.download([url])
        return True
    except Exception as e:
        print(f"    [yt-dlp] Failed: {e}")
        return False


# ── Fallback: extract direct URL and download with curl ─────────────────
def download_with_curl_fallback(url: str, output_path: str) -> bool:
    try:
        print(f"    [fallback] Extracting direct URL with yt-dlp...")
        with yt_dlp.YoutubeDL({"quiet": True, "skip_download": True}) as ydl:
            info = ydl.extract_info(url, download=False)
            direct_url = info.get("url") or (
                info.get("formats", [{}])[-1].get("url")
            )

        if not direct_url:
            print("    [fallback] Could not extract direct URL.")
            return False

        print(f"    [fallback] Downloading via curl...")
        result = subprocess.run(
            ["curl", "-L", "-o", output_path, direct_url],
            capture_output=True,
            text=True
        )
        if result.returncode == 0:
            return True
        else:
            print(f"    [fallback] curl failed: {result.stderr}")
            return False

    except Exception as e:
        print(f"    [fallback] Fallback failed: {e}")
        return False


# Download a single video 
def download_video(video: dict) -> dict:
    video_id    = video["id"]
    title       = video["title"]
    url         = video["url"]
    output_path = str(VIDEO_DIR / f"{video_id}.mp4")

    print(f"\n{'─'*55}")
    print(f"  Video : {video_id} — {title}")
    print(f"  URL   : {url}")
    print(f"{'─'*55}")

    # Skip if already downloaded
    if os.path.exists(output_path) and os.path.getsize(output_path) > 10_000:
        print(f"  ✅ Already downloaded, skipping.")
        return {**video, "status": "skipped", "file": output_path}

    # Try primary method
    success = download_with_ytdlp(url, output_path)

    # Try fallback if primary failed
    if not success:
        print(f"  ⚠️  Primary failed. Trying fallback...")
        success = download_with_curl_fallback(url, output_path)

    # Final result
    if success and os.path.exists(output_path):
        size_mb = os.path.getsize(output_path) / (1024 * 1024)
        print(f"  ✅ Downloaded successfully ({size_mb:.1f} MB)")
        return {**video, "status": "success", "file": output_path}
    else:
        print(f"  ❌ FAILED — manual intervention needed for: {url}")
        return {**video, "status": "failed", "file": None}


# Download all videos
def download_all_videos() -> list:
    with open(DATA_FILE, "r") as f:
        videos = json.load(f)

    print(f"\n🎬 Starting download of {len(videos)} videos...\n")

    results = []
    for video in videos:
        result = download_video(video)
        results.append(result)

    #  Summary 
    success = [r for r in results if r["status"] == "success"]
    skipped = [r for r in results if r["status"] == "skipped"]
    failed  = [r for r in results if r["status"] == "failed"]

    print(f"\n{'═'*55}")
    print(f"  DOWNLOAD SUMMARY")
    print(f"{'═'*55}")
    print(f"  ✅ Downloaded : {len(success)}")
    print(f"  ⏭️  Skipped    : {len(skipped)} (already existed)")
    print(f"  ❌ Failed     : {len(failed)}")

    if failed:
        print(f"\n  ⚠️  The following videos need manual attention:")
        for v in failed:
            print(f"     - {v['id']} : {v['url']}")

    print(f"{'═'*55}\n")

    # Save results log
    log_path = BASE_DIR / "data" / "download_log.json"
    with open(log_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  📄 Download log saved to: {log_path}\n")

    return results


#  Run directly
if __name__ == "__main__":
    download_all_videos()