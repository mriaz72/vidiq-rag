import json
import chromadb
from pathlib import Path
from sentence_transformers import SentenceTransformer
from chromadb.config import Settings

# Paths
BASE_DIR      = Path(__file__).resolve().parent.parent
CHUNKS_DIR    = BASE_DIR / "chunks"
VECTOR_DIR    = BASE_DIR / "vectorstore"
DATA_FILE     = BASE_DIR / "data" / "videos.json"
VECTOR_DIR.mkdir(exist_ok=True)

# Load embedding model 
print("Loading embedding model...")
embedder = SentenceTransformer("all-MiniLM-L6-v2")
print("Embedding model loaded\n")

#ChromaDB setup
chroma_client = chromadb.PersistentClient(
    path=str(VECTOR_DIR),
    settings=Settings(anonymized_telemetry=False)
)

COLLECTION_NAME = "vidiq_rag"


# Get or create collection 
def get_collection(reset: bool = False):
    if reset:
        try:
            chroma_client.delete_collection(COLLECTION_NAME)
            print(f"   Deleted existing collection: {COLLECTION_NAME}")
        except Exception:
            pass

    collection = chroma_client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"}  # cosine similarity
    )
    return collection


# Embed and store chunks for a single video 
def embed_video_chunks(video: dict, collection) -> dict:
    video_id    = video["id"]
    title       = video["title"]
    chunks_path = CHUNKS_DIR / f"{video_id}_chunks.json"

    print(f"\n{'─'*55}")
    print(f"  Video : {video_id} — {title}")
    print(f"{'─'*55}")

    if not chunks_path.exists():
        print(f"  Chunks file not found. Run chunker.py first")
        return {**video, "status": "failed", "embedded": 0}

    with open(chunks_path, encoding="utf-8") as f:
        chunks = json.load(f)

    if not chunks:
        print(f"  No chunks found in file")
        return {**video, "status": "failed", "embedded": 0}

    # Check if already embedded
    existing = collection.get(where={"video_id": video_id})
    if existing and len(existing["ids"]) > 0:
        print(f" Already embedded ({len(existing['ids'])} chunks), skipping.")
        return {**video, "status": "skipped",
                "embedded": len(existing["ids"])}

    print(f" Generating embeddings for {len(chunks)} chunks...")

    # Prepare data for ChromaDB
    ids         = []
    texts       = []
    metadatas   = []

    for chunk in chunks:
        ids.append(chunk["chunk_id"])
        texts.append(chunk["full_text"])
        metadatas.append({
            "video_id"          : chunk["video_id"],
            "title"             : chunk["title"],
            "url"               : chunk["url"],
            "start_sec"         : chunk["start_sec"],
            "end_sec"           : chunk["end_sec"],
            "start_fmt"         : chunk["start_fmt"],
            "end_fmt"           : chunk["end_fmt"],
            "timestamp"         : chunk["timestamp"],
            "transcript_text"   : chunk["transcript_text"][:500],
            "ocr_text"          : chunk.get("ocr_text", "")[:300],
            "visual_description": chunk.get("visual_description", "")[:300],
            "word_count"        : chunk["word_count"]
        })

    #Generate embeddings in batches
    batch_size  = 32
    embeddings  = []

    for i in range(0, len(texts), batch_size):
        batch      = texts[i:i + batch_size]
        batch_embs = embedder.encode(
            batch,
            show_progress_bar=False,
            normalize_embeddings=True
        )
        embeddings.extend(batch_embs.tolist())
        print(f"     Embedded batch {i//batch_size + 1}/"
              f"{(len(texts) + batch_size - 1)//batch_size}")

    # Store in ChromaDB
    collection.add(
        ids        = ids,
        documents  = texts,
        embeddings = embeddings,
        metadatas  = metadatas
    )

    print(f"  Embedded and stored {len(chunks)} chunks")

    return {**video, "status": "success", "embedded": len(chunks)}


# Embed all videos
def embed_all_videos(reset: bool = False) -> list:
    with open(DATA_FILE, "r") as f:
        videos = json.load(f)

    print(f"\n Starting embedding for {len(videos)} videos...\n")

    collection = get_collection(reset=reset)

    results = []
    for video in videos:
        result = embed_video_chunks(video, collection)
        results.append(result)

    success  = [r for r in results if r["status"] == "success"]
    skipped  = [r for r in results if r["status"] == "skipped"]
    failed   = [r for r in results if r["status"] == "failed"]
    total    = sum(r.get("embedded", 0) for r in success + skipped)

    print(f"\n{'═'*55}")
    print(f"  EMBEDDING SUMMARY")
    print(f"{'═'*55}")
    print(f"   Processed    : {len(success)} videos")
    print(f"   Skipped      : {len(skipped)} (already existed)")
    print(f"  Failed       : {len(failed)}")
    print(f"  Total chunks : {total}")
    print(f"  Stored in    : vectorstore/")
    print(f"{'═'*55}\n")

    # Quick search test
    print(" Running quick search test...")
    test_query     = "CRA payroll deductions consequences"
    test_embedding = embedder.encode(
        test_query,
        normalize_embeddings=True
    ).tolist()

    test_results = collection.query(
        query_embeddings=[test_embedding],
        n_results=2
    )

    if test_results and test_results["documents"][0]:
        print(f"  Search test passed")
        print(f"     Query   : '{test_query}'")
        for i, (doc, meta) in enumerate(zip(
            test_results["documents"][0],
            test_results["metadatas"][0]
        )):
            print(f"     Result {i+1}: {meta['title']} "
                  f"@ {meta['timestamp']}")
            print(f"              {doc[:100]}...")
    else:
        print(f" Search test returned no results")

    log_path = BASE_DIR / "data" / "embeddings_log.json"
    with open(log_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n Log saved to: data/embeddings_log.json\n")

    return results


# Run directly
if __name__ == "__main__":
    # Set reset=True to wipe and rebuild from scratch
    embed_all_videos(reset=False)