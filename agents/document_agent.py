import os
import tempfile
import uuid
from pathlib import Path
from typing import Iterable

import pdfplumber
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import (
    HnswAlgorithmConfiguration,
    SearchField,
    SearchFieldDataType,
    SearchIndex,
    SearchableField,
    SimpleField,
    VectorSearch,
    VectorSearchProfile,
)
from azure.search.documents.models import VectorizedQuery
from dotenv import load_dotenv
from fastembed import TextEmbedding
from groq import Groq

from utils.cosmos_logger import log_event

load_dotenv()

AZURE_SEARCH_INDEX = os.getenv("AZURE_SEARCH_INDEX", "documents")
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIMENSIONS = 384

# Conservative limits for the Azure App Service F1 free tier.
CHUNK_SIZE = 1_000
MAX_CHUNKS = 80
UPLOAD_BATCH_SIZE = 20

_embedder: TextEmbedding | None = None
_groq_client: Groq | None = None
_search_client: SearchClient | None = None
_index_initialized = False


def _required_env(name: str) -> str:
    """Return a required environment variable or raise a clear error."""
    value = os.getenv(name)
    if not value:
        raise RuntimeError(
            f"Missing environment variable: {name}. "
            "Add it in Azure App Service > Settings > Environment variables."
        )
    return value


def _get_cache_dir() -> str:
    """Use Azure persistent storage when configured, otherwise a local temp folder."""
    configured = os.getenv("FASTEMBED_CACHE_DIR")
    if configured:
        cache_dir = Path(configured)
    else:
        cache_dir = Path(tempfile.gettempdir()) / "fastembed_cache"

    cache_dir.mkdir(parents=True, exist_ok=True)
    return str(cache_dir)


def _get_embedder() -> TextEmbedding:
    """Lazy-load the ONNX embedding model only when RAG is used."""
    global _embedder

    if _embedder is None:
        _embedder = TextEmbedding(
            model_name=EMBEDDING_MODEL,
            cache_dir=_get_cache_dir(),
        )

    return _embedder


def _get_groq_client() -> Groq:
    """Create the Groq client only when an answer must be generated."""
    global _groq_client

    if _groq_client is None:
        _groq_client = Groq(api_key=_required_env("GROQ_API_KEY"))

    return _groq_client


def _get_search_client() -> SearchClient:
    """Create and reuse the Azure AI Search document client."""
    global _search_client

    if _search_client is None:
        endpoint = _required_env("AZURE_SEARCH_ENDPOINT")
        key = _required_env("AZURE_SEARCH_KEY")
        _search_client = SearchClient(
            endpoint=endpoint,
            index_name=AZURE_SEARCH_INDEX,
            credential=AzureKeyCredential(key),
        )

    return _search_client


def _create_or_update_index() -> None:
    """Create the Azure AI Search index if needed."""
    endpoint = _required_env("AZURE_SEARCH_ENDPOINT")
    key = _required_env("AZURE_SEARCH_KEY")

    index_client = SearchIndexClient(
        endpoint=endpoint,
        credential=AzureKeyCredential(key),
    )

    fields = [
        SimpleField(
            name="id",
            type=SearchFieldDataType.String,
            key=True,
        ),
        SearchableField(
            name="content",
            type=SearchFieldDataType.String,
        ),
        SimpleField(
            name="source",
            type=SearchFieldDataType.String,
            filterable=True,
        ),
        SearchField(
            name="embedding",
            type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
            searchable=True,
            vector_search_dimensions=EMBEDDING_DIMENSIONS,
            vector_search_profile_name="orion-vector-profile",
        ),
    ]

    vector_search = VectorSearch(
        algorithms=[
            HnswAlgorithmConfiguration(name="orion-hnsw"),
        ],
        profiles=[
            VectorSearchProfile(
                name="orion-vector-profile",
                algorithm_configuration_name="orion-hnsw",
            ),
        ],
    )

    index = SearchIndex(
        name=AZURE_SEARCH_INDEX,
        fields=fields,
        vector_search=vector_search,
    )

    index_client.create_or_update_index(index)


def _ensure_index() -> None:
    """Initialize Azure AI Search only when document features are first used."""
    global _index_initialized

    if not _index_initialized:
        _create_or_update_index()
        _index_initialized = True


def _embed_one(text: str) -> list[float]:
    """Generate one 384-dimensional embedding."""
    embeddings = _get_embedder().embed([text])
    return next(iter(embeddings)).tolist()


def _extract_chunks(file_path_or_bytes) -> list[str]:
    """Extract bounded text chunks from an uploaded PDF."""
    chunks: list[str] = []

    with pdfplumber.open(file_path_or_bytes) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if not text:
                continue

            normalized = " ".join(text.split())
            page_chunks = [
                normalized[start : start + CHUNK_SIZE]
                for start in range(0, len(normalized), CHUNK_SIZE)
            ]
            chunks.extend(page_chunks)

            if len(chunks) >= MAX_CHUNKS:
                return chunks[:MAX_CHUNKS]

    return chunks


def _batched(items: list[dict], batch_size: int) -> Iterable[list[dict]]:
    """Yield small upload batches to limit memory and request size."""
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def ingest_pdf(file_path_or_bytes, source_name: str = "uploaded_pdf") -> int:
    """Extract, embed, and upload PDF chunks to Azure AI Search."""
    _ensure_index()

    chunks = _extract_chunks(file_path_or_bytes)
    if not chunks:
        return 0

    # FastEmbed returns a generator, keeping memory usage lower than PyTorch.
    vectors = _get_embedder().embed(chunks)

    documents: list[dict] = []
    for chunk, vector in zip(chunks, vectors):
        documents.append(
            {
                "id": str(uuid.uuid4()),
                "content": chunk,
                "source": source_name,
                "embedding": vector.tolist(),
            }
        )

    client = _get_search_client()

    for batch in _batched(documents, UPLOAD_BATCH_SIZE):
        upload_results = client.upload_documents(documents=batch)
        failed = [result for result in upload_results if not result.succeeded]
        if failed:
            failed_keys = ", ".join(result.key for result in failed)
            raise RuntimeError(
                f"Azure AI Search failed to upload document keys: {failed_keys}"
            )

    return len(documents)


def query(question: str, workflow_id: str = "standalone") -> str:
    """Run hybrid keyword and vector retrieval, then answer with Groq."""
    _ensure_index()

    question = question.strip()
    if not question:
        return "Please provide a document question."

    vector_query = VectorizedQuery(
        vector=_embed_one(question),
        k_nearest_neighbors=3,
        fields="embedding",
    )

    results = _get_search_client().search(
        search_text=question,
        vector_queries=[vector_query],
        select=["content", "source"],
        top=3,
    )

    retrieved = list(results)
    docs = [item["content"] for item in retrieved if item.get("content")]

    if not docs:
        return "No relevant document content was found. Please upload a PDF first."

    context = "\n\n".join(docs)

    prompt = f"""You are a document analysis assistant.
Use only the supplied document excerpts to answer the question accurately.
If the answer is not present in the excerpts, state that clearly.

Document excerpts:
{context}

Question:
{question}
"""

    response = _get_groq_client().chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=1024,
        temperature=0.2,
    )

    answer = response.choices[0].message.content or "No answer was generated."
    log_event(workflow_id, "DocumentAgent", question, answer)
    return answer
