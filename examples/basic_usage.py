"""
Simple example showing how to use the Hybrid Retriever with document management
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bm25_chroma.hybrid_retriever import HybridRetriever

def main():
    # Initialize retriever
    retriever = HybridRetriever(
        chroma_path="./test_db",
        collection_name="simple_docs"
    )
    
    # Add some documents
    documents = [
        "Machine learning is a powerful tool for data analysis and prediction.",
        "Natural language processing helps computers understand human language.",
        "Deep learning neural networks can learn complex patterns from data.",
        "Computer vision enables machines to interpret and analyze visual information.",
        "Reinforcement learning trains agents through trial and error."
    ]
    
    doc_ids = ["ml_doc", "nlp_doc", "dl_doc", "cv_doc", "rl_doc"]
    
    # Add documents in unified mode
    stats = retriever.add_documents_batch(
        documents,
        doc_ids=doc_ids,
        mode="unified",
        chroma_batch_size=8,
        show_progress=True
    )
    
    print(f"\nAdded {stats['total_documents']} documents in {stats['total_time_seconds']:.2f}s")
    
    # Show initial system stats
    print(f"\nInitial system stats:")
    initial_stats = retriever.get_system_stats()
    for key, value in initial_stats.items():
        print(f"  {key}: {value}")
    
    # Test search before deletion
    query = "machine learning data analysis"
    print(f"\nSearching for: '{query}' (before deletion)")
    results = retriever.hybrid_search(query, top_k=3)
    
    print("Results before deletion:")
    for i, (doc_id, score, metadata) in enumerate(results, 1):
        text = metadata['text'][:60]
        print(f"{i}. {doc_id}: {score:.3f} - {text}...")
    
    # Delete a document
    print(f"\nDeleting document: 'cv_doc'")
    retriever.remove_document("cv_doc")
    
    # Show stats after deletion
    after_delete_stats = retriever.get_system_stats()
    print(f"Documents after deletion: {after_delete_stats['chunks']} (was {initial_stats['chunks']})")
    
    # Add a new document
    new_document = "Quantum computing leverages quantum mechanics for advanced computation."
    print(f"\nAdding new document: 'quantum_doc'")
    retriever.add_documents_batch(
        [new_document], 
        doc_ids=["quantum_doc"],
        show_progress=False
    )
    
    # Show final stats
    final_stats = retriever.get_system_stats()
    print(f"Documents after addition: {final_stats['chunks']}")
    
    # Test search after changes
    print(f"\nSearching for: 'quantum computation' (after changes)")
    quantum_results = retriever.hybrid_search("quantum computation", top_k=3)
    
    print("Results after document management:")
    for i, (doc_id, score, metadata) in enumerate(quantum_results, 1):
        text = metadata['text'][:60]
        print(f"{i}. {doc_id}: {score:.3f} - {text}...")
    
    # Batch deletion example
    print(f"\nBatch deleting: ['nlp_doc', 'rl_doc']")
    retriever.remove_documents_batch(["nlp_doc", "rl_doc"])
    
    final_final_stats = retriever.get_system_stats()
    print(f"Final document count: {final_final_stats['chunks']}")

if __name__ == "__main__":
    main()