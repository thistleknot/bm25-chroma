from bm25_chroma import HybridRetriever, BM25

# Initialize
retriever = HybridRetriever(
    chroma_path="./my_db",
    collection_name="my_docs"
)

# Add documents
documents = [
    "Machine learning helps analyze data patterns.",
    "Natural language processing understands human text.",
    "Deep learning uses neural networks for complex tasks."
]

retriever.add_documents_batch(
    documents,
    doc_ids=["doc1", "doc2", "doc3"],  # Optional: auto-generated if not provided
    mode="unified",  # or "sequential"
    show_progress=True
)

# Search
results = retriever.hybrid_search("machine learning", top_k=5)
for doc_id, score, metadata in results:
    print(f"{doc_id}: {score:.3f} - {metadata['text'][:100]}...")

# Document management
retriever.remove_document("doc1")  # Remove single document
retriever.remove_documents_batch(["doc2", "doc3"])  # Batch removal

# Add new documents
new_docs = ["Quantum computing leverages quantum mechanics."]
retriever.add_documents_batch(new_docs, doc_ids=["quantum_doc"])