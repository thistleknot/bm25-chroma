"""
Simple example showing how to use the Enhanced Hybrid Retriever
"""

from hybrid_retriever import EnhancedHybridRetriever

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
    
    # Add documents in unified mode
    stats = retriever.add_documents_batch(
        documents,
        mode="unified",
        chroma_batch_size=8,
        show_progress=True
    )
    
    print(f"\nAdded {stats['total_documents']} documents in {stats['total_time_seconds']:.2f}s")
    
    # Test search
    query = "machine learning data analysis"
    print(f"\n🔍 Searching for: '{query}'")
    
    # Hybrid search
    results = retriever.hybrid_search(query, top_k=3)
    
    print("\nTop results:")
    for i, (doc_id, score, metadata) in enumerate(results, 1):
        text = metadata['text']
        print(f"{i}. {doc_id}: {score:.3f}")
        print(f"   {text}")
    
    # Show system stats
    print(f"\nSystem stats:")
    stats = retriever.get_system_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()