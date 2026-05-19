import os
import sys
import time
import faiss
import numpy as np
import nltk
from pathlib import Path


from google import genai
from google.genai import types
from huggingface_hub import InferenceClient
from nltk.tokenize import sent_tokenize
from dotenv import load_dotenv


load_dotenv()

# ==========================================================
# HARD-CODED TOKENS
# ==========================================================

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
HF_TOKEN = os.getenv("HF_TOKEN")


# ==========================================================
# CONFIGURATION
# ==========================================================

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_FOLDER = BASE_DIR / "data"

VECTOR_STORE_DIR = BASE_DIR / "vector_store"
FAISS_INDEX_PATH = VECTOR_STORE_DIR / "faiss.index"
CHUNKS_PATH = VECTOR_STORE_DIR / "chunks.npy"

# Hugging Face cloud embedding model
HF_EMBEDDING_MODEL = "ibm-granite/granite-embedding-97m-multilingual-r2"

# You can also try:
# HF_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Gemini cloud LLM
GEMINI_MODEL = "gemini-2.5-flash"

TOP_K = 3
SIMILARITY_THRESHOLD = 0.40

# Start with 1 to avoid connection problems.
# Later you can try 4 or 8.
BATCH_SIZE = 1


# ==========================================================
# CLIENTS
# ==========================================================

gemini_client = genai.Client(api_key=GEMINI_API_KEY)

hf_client = InferenceClient(
    provider="hf-inference",
    api_key=HF_TOKEN
)


# ==========================================================
# NLTK SETUP
# ==========================================================

def setup_nltk():
    nltk.download("punkt", quiet=True)
    nltk.download("punkt_tab", quiet=True)


# ==========================================================
# LOAD DOCUMENTS
# ==========================================================

def load_documents(folder=DATA_FOLDER):
    """
    Load .txt files from the data folder and split them into text chunks.
    """

    if not os.path.exists(folder):
        raise FileNotFoundError(
            f"Folder '{folder}' does not exist. Create it and put .txt files inside."
        )

    chunks = []

    for file_name in os.listdir(folder):
        if file_name.endswith(".txt"):
            file_path = os.path.join(folder, file_name)

            with open(file_path, "r", encoding="utf-8") as file:
                text = file.read()

            sentences = sent_tokenize(text)

            for sentence in sentences:
                sentence = sentence.strip()

                if sentence:
                    chunks.append(sentence)

    if not chunks:
        raise ValueError(
            f"No text found. Make sure the '{folder}' folder contains .txt files."
        )

    print(f"Loaded {len(chunks)} text chunks.")
    return chunks


# ==========================================================
# HUGGING FACE CLOUD EMBEDDINGS
# ==========================================================

def normalize_embedding_output(raw_output, expected_count):
    """
    Converts Hugging Face embedding output into a clean 2D numpy array.

    Final shape:
        [number_of_texts, embedding_dimension]
    """

    arr = np.array(raw_output, dtype="float32")

    # Case 1:
    # Single embedding:
    # [embedding_dimension]
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)

    # Case 2:
    # Batch embeddings:
    # [batch_size, embedding_dimension]
    elif arr.ndim == 2:
        if arr.shape[0] == expected_count:
            pass

        # Token embeddings for one input:
        # [tokens, embedding_dimension]
        elif expected_count == 1:
            arr = arr.mean(axis=0, keepdims=True)

        else:
            raise ValueError(
                f"Unexpected 2D embedding shape: {arr.shape}, "
                f"expected_count={expected_count}"
            )

    # Case 3:
    # Token embeddings for batch:
    # [batch_size, tokens, embedding_dimension]
    elif arr.ndim == 3:
        arr = arr.mean(axis=1)

    else:
        raise ValueError(f"Unexpected embedding dimensions: {arr.ndim}")

    if arr.shape[0] != expected_count:
        raise ValueError(
            f"Embedding count mismatch. Expected {expected_count}, got {arr.shape[0]}"
        )

    return arr.astype("float32")


def hf_feature_extraction_with_retries(inputs, expected_count, max_retries=5):
    """
    Calls Hugging Face cloud embedding model with retries.

    inputs can be:
    - string
    - list of strings
    """

    for attempt in range(1, max_retries + 1):
        try:
            result = hf_client.feature_extraction(
                inputs,
                model=HF_EMBEDDING_MODEL
            )

            embeddings = normalize_embedding_output(
                raw_output=result,
                expected_count=expected_count
            )

            return embeddings

        except Exception as e:
            print(
                f"Hugging Face embedding failed. Attempt {attempt}/{max_retries}")
            print("Error:", e)

            if attempt == max_retries:
                raise

            wait_seconds = attempt * 3
            print(f"Retrying in {wait_seconds} seconds...")
            time.sleep(wait_seconds)


def embed_texts_with_huggingface(texts, batch_size=BATCH_SIZE):
    """
    Creates document embeddings using Hugging Face cloud inference.
    """

    all_embeddings = []
    total_batches = (len(texts) + batch_size - 1) // batch_size

    for start in range(0, len(texts), batch_size):
        batch = texts[start:start + batch_size]

        current_batch = start // batch_size + 1
        print(f"Embedding batch {current_batch}/{total_batches}...")

        embeddings = hf_feature_extraction_with_retries(
            inputs=batch,
            expected_count=len(batch)
        )

        all_embeddings.append(embeddings)

    final_embeddings = np.vstack(all_embeddings).astype("float32")

    print(f"Created document embeddings. Shape: {final_embeddings.shape}")

    return final_embeddings


def embed_query_with_huggingface(query):
    """
    Creates one query embedding using Hugging Face cloud inference.
    """

    embedding = hf_feature_extraction_with_retries(
        inputs=query,
        expected_count=1
    )

    return embedding.astype("float32")


# ==========================================================
# FAISS VECTOR SEARCH
# ==========================================================

def create_faiss_index(embeddings):
    """
    Creates FAISS index.

    We normalize vectors and use inner product.
    This behaves like cosine similarity.
    """

    faiss.normalize_L2(embeddings)

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)

    print(f"FAISS index created with {index.ntotal} vectors.")

    return index


def retrieve(query, index, chunks, k=TOP_K, threshold=SIMILARITY_THRESHOLD):
    """
    Embeds the user question with Hugging Face and searches FAISS.
    Returns only chunks whose cosine similarity exceeds the threshold.
    """

    query_embedding = embed_query_with_huggingface(query)

    faiss.normalize_L2(query_embedding)

    scores, indexes = index.search(query_embedding, k)

    print("\nFAISS scores:", scores)
    print("FAISS indexes:", indexes)

    retrieved_chunks = []

    for score, idx in zip(scores[0], indexes[0]):
        if idx == -1:
            continue
        if float(score) < threshold:
            print(f"  Skipping chunk {idx} — score {score:.3f} below threshold {threshold}")
            continue
        retrieved_chunks.append(chunks[idx])

    return retrieved_chunks


# ==========================================================
# GEMINI LLM
# ==========================================================

def ask_gemini(context, question):
    """
    Gemini is the LLM.
    Hugging Face is only used for embeddings.
    """
    prompt = f"""You are an expert assistant on the Red Hot Chili Peppers (RHCP).
Answer the user's question based ONLY on the numbered context excerpts below, retrieved from the RHCP knowledge base.

Rules:
- Answer using ONLY information explicitly stated in the context excerpts.
- Do not invent facts, names, dates, albums, or events not present in the context.
- If the context does not contain sufficient information to answer, respond with exactly:
  "I don't have enough information in my knowledge base to answer that."
- Do not speculate, infer beyond what is stated, or draw on outside knowledge.
- Be concise and accurate.

Context excerpts:
{context}

Question: {question}

Answer:"""

    response = gemini_client.models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=0.3,
            max_output_tokens=500,
            thinking_config=types.ThinkingConfig(
                thinking_budget=0
            )
        )
    )

    return response.text.strip()


# ==========================================================
# VECTOR STORE — SAVE / LOAD
# ==========================================================

def save_vector_store(index, chunks):
    VECTOR_STORE_DIR.mkdir(exist_ok=True)
    faiss.write_index(index, str(FAISS_INDEX_PATH))
    np.save(CHUNKS_PATH, np.array(chunks, dtype=object))
    print("Vector store saved to disk.")


def load_vector_store():
    if not FAISS_INDEX_PATH.exists() or not CHUNKS_PATH.exists():
        return None, None
    index = faiss.read_index(str(FAISS_INDEX_PATH))
    chunks = np.load(CHUNKS_PATH, allow_pickle=True).tolist()
    print(f"Vector store loaded from disk ({len(chunks)} chunks).")
    return index, chunks


# ==========================================================
# LAZY-LOADED STATE (built once per process)
# ==========================================================

_index = None
_chunks = None


def _init():
    global _index, _chunks
    if _index is not None:
        return
    setup_nltk()
    _index, _chunks = load_vector_store()
    if _index is not None:
        return
    print("Building vector store from scratch...")
    chunks = load_documents(DATA_FOLDER)
    embeddings = embed_texts_with_huggingface(chunks)
    index = create_faiss_index(embeddings)
    save_vector_store(index, chunks)
    _index = index
    _chunks = chunks


# ==========================================================
# PUBLIC API (called by RHCP_app.py)
# ==========================================================

def ask_question(question):
    question = question.strip()
    if not question:
        return {"answer": "Please enter a question.", "sources": []}

    _init()

    top_chunks = retrieve(question, _index, _chunks, k=TOP_K)

    if not top_chunks:
        return {
            "answer": (
                "I couldn't find relevant information in the RHCP knowledge base "
                "for that question. Try asking about the band's history, albums, "
                "members, or songs."
            ),
            "sources": [],
        }

    numbered_context = "\n\n".join(
        f"[{i+1}] {chunk}" for i, chunk in enumerate(top_chunks)
    )
    answer = ask_gemini(numbered_context, question)
    return {"answer": answer, "sources": top_chunks}


# ==========================================================
# MAIN APP
# ==========================================================

def main():
    setup_nltk()

    print("Loading documents...")
    chunks = load_documents(DATA_FOLDER)

    print("\nCreating Hugging Face cloud embeddings...")
    document_embeddings = embed_texts_with_huggingface(chunks)

    print("\nCreating FAISS index...")
    index = create_faiss_index(document_embeddings)

    print("\nRAG system is ready.")
    print("Embeddings: Hugging Face cloud")
    print("Vector search: FAISS local")
    print("LLM: Gemini cloud")
    print("Type 'exit' to quit.")

    while True:
        question = input("\nAsk something: ").strip()

        if question.lower() == "exit":
            print("Goodbye.")
            break

        if not question:
            print("Please enter a real question.")
            continue

        top_chunks = retrieve(
            query=question,
            index=index,
            chunks=chunks,
            k=TOP_K
        )

        context = "\n".join(top_chunks)

        print("\nRetrieved Context:")
        print(context)

        answer = ask_gemini(context, question)

        print("\nGemini Answer:")
        print(answer)


if __name__ == "__main__":
    main()
