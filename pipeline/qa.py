import json
import os
from pathlib import Path
from groq import Groq
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
from dotenv import load_dotenv

load_dotenv()

#Paths 
BASE_DIR   = Path(__file__).resolve().parent.parent
VECTOR_DIR = BASE_DIR / "vectorstore"
OUTPUTS    = BASE_DIR / "outputs"
OUTPUTS.mkdir(exist_ok=True)

# Groq setup
client    = Groq(api_key=os.getenv("GROQ_API_KEY_2"))
LLM_MODEL = "llama-3.3-70b-versatile"

#  Embedding model 
print("Loading embedding model...")
embedder = SentenceTransformer("all-MiniLM-L6-v2")
print("✅ Embedding model loaded\n")

# ChromaDB
chroma_client = chromadb.PersistentClient(
    path=str(VECTOR_DIR),
    settings=Settings(anonymized_telemetry=False)
)
collection = chroma_client.get_collection("vidiq_rag")

#System prompt
SYSTEM_PROMPT = """You are a knowledge assistant built from video content.

STRICT RULES:
1. Answer ONLY from the context provided below.
2. Do NOT use your general knowledge under any circumstances.
3. If the answer is not found in the context, respond with exactly:
   "I don't know based on the provided video content."
4. Every answer MUST cite the video title and timestamp.
5. Be concise and direct.

SEMANTIC MATCHING RULES:
- If the user asks about "penalties" look for consequences, enforcement,
  actions, fees, charges, freezing, garnishing, seizing in the context.
- If the user asks about "rights" look for legal protections, what you
  can do, options available in the context.
- If the user asks about "options" look for choices, alternatives,
  solutions, ways to handle in the context.
- If the user asks about "cost" look for fees, interest, charges,
  amounts, rates in the context.
- Generally: look for the MEANING of the question, not just exact words.
- If the context contains information that answers the question even with
  different wording, USE IT to answer."""


#Retrieve relevant chunks
def retrieve(query: str, n_results: int = 4) -> list:
    emb = embedder.encode(
        query,
        normalize_embeddings=True
    ).tolist()

    results = collection.query(
        query_embeddings=[emb],
        n_results=n_results
    )

    chunks = []
    if results and results["documents"][0]:
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0]
        ):
            chunks.append({
                "text"    : doc,
                "metadata": meta,
                "score"   : round(1 - dist, 4)
            })

    return chunks


#Generate answer
def generate_answer(query: str, chunks: list) -> str:
    if not chunks:
        return "I don't know based on the provided video content."

    context = "\n\n".join([
        f"--- SOURCE {i+1} ---\n"
        f"Video : {c['metadata']['title']}\n"
        f"Time  : {c['metadata']['timestamp']}\n"
        f"URL   : {c['metadata']['url']}\n"
        f"Content:\n{c['text'][:400]}"
        for i, c in enumerate(chunks)
    ])

    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"CONTEXT:\n{context}\n\nQUESTION: {query}"
            }
        ],
        temperature=0.0,
        max_tokens=400
    )

    return response.choices[0].message.content.strip()


# Full answer pipeline 
def answer(query: str, n_results: int = 4) -> dict:
    # Step 1: Retrieve
    chunks = retrieve(query, n_results)

    # Step 2: Generate
    answer_text = generate_answer(query, chunks)

    # Step 3: Build sources
    sources = []
    seen    = set()
    for chunk in chunks:
        meta = chunk["metadata"]
        key  = f"{meta['title']}_{meta['timestamp']}"
        if key not in seen:
            seen.add(key)
            sources.append({
                "title"    : meta["title"],
                "url"      : meta["url"],
                "timestamp": meta["timestamp"],
                "score"    : chunk["score"]
            })

    return {
        "query"  : query,
        "answer" : answer_text,
        "sources": sources,
        "chunks" : chunks
    }


# Pretty print result
def print_result(result: dict):
    print(f"\n{'═'*60}")
    print(f"  QUESTION : {result['query']}")
    print(f"{'═'*60}")
    print(f"\n  ANSWER:\n  {result['answer']}\n")
    print(f"{'─'*60}")
    print(f"  SOURCES:")
    for i, src in enumerate(result["sources"]):
        print(f"\n  [{i+1}] {src['title']}")
        print(f"       Timestamp : {src['timestamp']}")
        print(f"       URL       : {src['url']}")
        print(f"       Relevance : {src['score']}")
    print(f"{'═'*60}\n")


#Run 10 sample Q&A + 2 no-answer tests
def run_sample_qa():
    questions = [
        # Answerable from videos
        "What are the immediate consequences of unpaid payroll deductions?",
        "What should I do if I am behind on payroll source deduction payments?",
        "Can I negotiate a payment plan with the CRA?",
        "Do CRA interest and fees keep accruing while negotiating?",
        "What are my legal rights when dealing with the CRA?",
        "How does payroll debt affect my ability to get financing?",
        "How can I walk away from my payroll debt?",
        "What happens if I ignore CRA payroll debt?",
        "What penalties does the CRA apply for unpaid payroll deductions?",

        # No-answer tests
        "What is the corporate tax rate in the United States?",
        "How do I file a personal income tax return in Australia?"
    ]

    all_results = []

    print(f"\n Running {len(questions)} sample Q&A tests...\n")

    for i, q in enumerate(questions):
        print(f"  [{i+1}/{len(questions)}] {q[:60]}...")
        result = answer(q)
        print_result(result)
        all_results.append({
            "question": result["query"],
            "answer"  : result["answer"],
            "sources" : result["sources"]
        })

    # Save JSON
    output_path = OUTPUTS / "sample_qa.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    # Save readable TXT
    readable_path = OUTPUTS / "sample_qa_readable.txt"
    with open(readable_path, "w", encoding="utf-8") as f:
        for r in all_results:
            f.write(f"{'═'*60}\n")
            f.write(f"Q: {r['question']}\n")
            f.write(f"{'─'*60}\n")
            f.write(f"A: {r['answer']}\n\n")
            f.write(f"SOURCES:\n")
            for src in r["sources"]:
                f.write(f"  - {src['title']} @ {src['timestamp']}\n")
                f.write(f"    {src['url']}\n")
            f.write(f"\n")

    print(f"\n Sample Q&A complete")
    print(f"   Saved to: outputs/sample_qa.json")
    print(f"   Saved to: outputs/sample_qa_readable.txt\n")

    return all_results


# Interactive mode
def interactive():
    print("\n VidIQ-RAG Interactive Q&A")
    print("   Type your question and press Enter")
    print("   Type 'quit' to exit\n")

    while True:
        query = input("Your question: ").strip()
        if query.lower() in ["quit", "exit", "q"]:
            print("Goodbye!")
            break
        if not query:
            continue
        result = answer(query)
        print_result(result)


# Run directly
if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "interactive":
        interactive()
    else:
        run_sample_qa()