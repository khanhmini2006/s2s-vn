import os
from typing import List

import chromadb
from chromadb.utils import embedding_functions


class RAGService:
    def __init__(self, persist_directory="./chroma_db"):
        self.persist_directory = persist_directory
        self.client = chromadb.PersistentClient(path=self.persist_directory)
        
        # Use a multilingual embedding model that supports Vietnamese well.
        # BAAI/bge-m3 or intfloat/multilingual-e5-small are good choices.
        # Here we use multilingual-e5-small as it's lighter for realtime.
        model_name = "intfloat/multilingual-e5-small"
        self.embedding_func = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=model_name)
        
        self.collection = self.client.get_or_create_collection(
            name="knowledge_base",
            embedding_function=self.embedding_func
        )

    def add_document(self, text: str, doc_id: str, metadata: dict = None):
        """Chunk a document and add it to the vector database."""
        chunks = self._chunk_text(text)
        
        ids = [f"{doc_id}_chunk_{i}" for i in range(len(chunks))]
        metadatas = [metadata for _ in range(len(chunks))] if metadata else None
        
        self.collection.add(
            documents=chunks,
            ids=ids,
            metadatas=metadatas
        )
        return len(chunks)

    def search(self, query: str, top_k: int = 3) -> List[str]:
        """Search the vector database for the most relevant chunks."""
        results = self.collection.query(
            query_texts=[query],
            n_results=top_k
        )
        
        if not results['documents'] or not results['documents'][0]:
            return []
            
        return results['documents'][0]

    def list_documents(self) -> List[dict]:
        """List all documents grouped by doc_id."""
        all_data = self.collection.get(include=["metadatas"])
        if not all_data["ids"]:
            return []

        # Group chunks by doc_id prefix (e.g. "doc_abc12345_chunk_0" → "doc_abc12345")
        docs: dict[str, dict] = {}
        for chunk_id, meta in zip(all_data["ids"], all_data["metadatas"] or [{}] * len(all_data["ids"])):
            # Extract doc_id: everything before "_chunk_"
            parts = chunk_id.rsplit("_chunk_", 1)
            doc_id = parts[0] if len(parts) == 2 else chunk_id
            if doc_id not in docs:
                docs[doc_id] = {
                    "doc_id": doc_id,
                    "filename": (meta or {}).get("filename", "unknown"),
                    "chunks": 0,
                }
            docs[doc_id]["chunks"] += 1

        return list(docs.values())

    def delete_document(self, doc_id: str) -> int:
        """Delete all chunks belonging to a document. Returns number of chunks deleted."""
        all_data = self.collection.get()
        to_delete = [cid for cid in all_data["ids"] if cid.startswith(f"{doc_id}_chunk_")]
        if not to_delete:
            # Maybe exact match (single-chunk doc)
            to_delete = [cid for cid in all_data["ids"] if cid == doc_id]
        if to_delete:
            self.collection.delete(ids=to_delete)
        return len(to_delete)

    def _chunk_text(self, text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
        """Simple text chunker based on characters."""
        chunks = []
        start = 0
        text_length = len(text)
        
        while start < text_length:
            end = start + chunk_size
            chunks.append(text[start:end])
            start += chunk_size - overlap
            
        return chunks

# Singleton instance
rag_service = RAGService()
