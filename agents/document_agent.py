import os
import uuid
import pdfplumber
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import (
    SearchIndex, SimpleField, SearchableField,
    SearchField, SearchFieldDataType, VectorSearch,
    HnswAlgorithmConfiguration, VectorSearchProfile
)
from azure.core.credentials import AzureKeyCredential
from sentence_transformers import SentenceTransformer
from groq import Groq
from dotenv import load_dotenv
from utils.cosmos_logger import log_event

load_dotenv()

_endpoint = os.getenv("AZURE_SEARCH_ENDPOINT")
_key = os.getenv("AZURE_SEARCH_KEY")
_index_name = os.getenv("AZURE_SEARCH_INDEX", "documents")
_groq = Groq(api_key=os.getenv("GROQ_API_KEY"))
_embedder = SentenceTransformer("all-MiniLM-L6-v2")
_credential = AzureKeyCredential(_key)

def _create_index():
    index_client = SearchIndexClient(endpoint=_endpoint, credential=_credential)
    fields = [
        SimpleField(name="id", type=SearchFieldDataType.String, key=True),
        SearchableField(name="content", type=SearchFieldDataType.String),
        SimpleField(name="source", type=SearchFieldDataType.String, filterable=True),
        SearchField(
            name="embedding",
            type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
            searchable=True,
            vector_search_dimensions=384,
            vector_search_profile_name="my-profile"
        )
    ]
    vector_search = VectorSearch(
        algorithms=[HnswAlgorithmConfiguration(name="my-hnsw")],
        profiles=[VectorSearchProfile(name="my-profile", algorithm_configuration_name="my-hnsw")]
    )
    index = SearchIndex(name=_index_name, fields=fields, vector_search=vector_search)
    index_client.create_or_update_index(index)

def _get_search_client():
    return SearchClient(endpoint=_endpoint, index_name=_index_name, credential=_credential)

_create_index()

def ingest_pdf(file_path_or_bytes, source_name="uploaded_pdf"):
    chunks = []
    if hasattr(file_path_or_bytes, "read"):
        with pdfplumber.open(file_path_or_bytes) as pdf:
            for page in pdf.pages:
                text = page.extract_text()
                if text:
                    chunks += [text[i:i+500] for i in range(0, len(text), 500)]
    else:
        with pdfplumber.open(file_path_or_bytes) as pdf:
            for page in pdf.pages:
                text = page.extract_text()
                if text:
                    chunks += [text[i:i+500] for i in range(0, len(text), 500)]
    if not chunks:
        return 0
    client = _get_search_client()
    documents = []
    for chunk in chunks:
        embedding = _embedder.encode([chunk])[0].tolist()
        documents.append({
            "id": str(uuid.uuid4()),
            "content": chunk,
            "source": source_name,
            "embedding": embedding
        })
    client.upload_documents(documents=documents)
    return len(chunks)

def query(question, workflow_id="standalone"):
    q_embedding = _embedder.encode([question])[0].tolist()
    client = _get_search_client()
    from azure.search.documents.models import VectorizedQuery
    vector_query = VectorizedQuery(
        vector=q_embedding,
        k_nearest_neighbors=3,
        fields="embedding"
    )
    results = client.search(
        search_text=question,
        vector_queries=[vector_query],
        select=["content", "source"],
        top=3
    )
    docs = [r["content"] for r in results]
    if not docs:
        return "No relevant document content found. Please upload a PDF first."
    context = "\n\n".join(docs)
    prompt = f"""You are a document analysis assistant.
Use the following document excerpts to answer the question accurately.

Document excerpts:
{context}

Question: {question}

Answer based only on the provided excerpts. If the answer is not in the excerpts, say so."""
    response = _groq.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=1024
    )
    answer = response.choices[0].message.content
    log_event(workflow_id, "DocumentAgent", question, answer)
    return answer