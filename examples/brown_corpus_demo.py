"""
Brown Corpus demo with document management workflow
"""

import nltk
from nltk.corpus import brown
from nltk.tokenize.treebank import TreebankWordDetokenizer
import re
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from bm25_chroma.hybrid_retriever import HybridRetriever

# Download Brown corpus if needed
nltk.download('brown', quiet=True)

def get_brown_docs(num_docs=30):
    """Extract clean document texts from Brown corpus"""
    fileids = brown.fileids()[:num_docs]
    docs = []
    detokenizer = TreebankWordDetokenizer()
    
    for fileid in fileids:
        paragraphs = brown.paras(fileid)
        doc_paragraphs = []
        
        for paragraph in paragraphs:
            para_lines = []
            
            for sentence in paragraph:
                # Convert quotes
                processed_sent = []
                for token in sentence:
                    if token == '``':
                        processed_sent.append('"')
                    elif token == "''":
                        processed_sent.append('"')
                    else:
                        processed_sent.append(token)
                
                # Detokenize and clean
                detokenized_line = detokenizer.detokenize(processed_sent)
                
                # Fix spacing issues
                detokenized_line = re.sub(r'\s+([.!?])\s+', r'\1 ', detokenized_line)
                detokenized_line = re.sub(r'"\s+([.!?])', r'"\1', detokenized_line)
                detokenized_line = re.sub(r'"\s+([^"]+?)\s+"', r'"\1"', detokenized_line)
                detokenized_line = re.sub(r'^\s*"\s+', r'"', detokenized_line)
                detokenized_line = re.sub(r'\s+"\s*$', r'"', detokenized_line)
                detokenized_line = re.sub(r'([.!?])([A-Z])', r'\1 \2', detokenized_line)
                detokenized_line = re.sub(r'([,:;])([A-Za-z])', r'\1 \2', detokenized_line)
                detokenized_line = re.sub(r'\s+', ' ', detokenized_line).strip()
                
                para_lines.append(detokenized_line)
            
            paragraph_text = ' '.join(para_lines)
            doc_paragraphs.append(paragraph_text)
        
        doc_text = '\n\n'.join(doc_paragraphs)
        if len(doc_text.strip()) >= 100:  # Filter short docs
            docs.append(doc_text)
    
    return docs[:num_docs]

def main():
    print("Brown Corpus Hybrid Retriever Demo with Document Management")
    
    # Get Brown corpus documents
    print("Loading Brown corpus...")
    brown_docs = get_brown_docs(30)
    print(f"Loaded {len(brown_docs)} documents")
    
    # Initialize retriever
    retriever = HybridRetriever(
        chroma_path="./brown_db",
        collection_name="brown_corpus_management"
    )
    
    # Add initial documents
    print("\n" + "="*60)
    print("ADDING INITIAL DOCUMENTS")
    print("="*60)
    
    doc_ids = [f"brown_{i:04d}" for i in range(len(brown_docs))]
    
    stats = retriever.add_documents_batch(
        brown_docs,
        doc_ids=doc_ids,
        mode="unified",
        chroma_batch_size=8,
        show_progress=True
    )
    
    print(f"Added {stats['total_documents']} documents in {stats['total_time_seconds']:.2f}s")
    
    # Show initial system stats
    initial_stats = retriever.get_system_stats()
    print(f"Initial document count: {initial_stats['chunks']}")
    
    # Test search before deletion
    print("\n" + "="*60)
    print("SEARCH BEFORE DOCUMENT MANAGEMENT")
    print("="*60)
    
    query = "government political"
    print(f"Searching for: '{query}'")
    
    initial_results = retriever.hybrid_search(query, top_k=5)
    print("Initial search results:")
    for i, (doc_id, score, metadata) in enumerate(initial_results, 1):
        text = metadata.get('text', '')[:100]
        print(f"  {i}. {doc_id}: {score:.3f} - {text}...")
    
    # Document management workflow
    print("\n" + "="*60)
    print("DOCUMENT MANAGEMENT WORKFLOW")
    print("="*60)
    
    # Remove some documents
    docs_to_remove = doc_ids[:5]  # Remove first 5 documents
    print(f"Removing {len(docs_to_remove)} documents: {docs_to_remove}")
    
    retriever.remove_documents_batch(docs_to_remove)
    
    after_deletion_stats = retriever.get_system_stats()
    print(f"Documents after deletion: {after_deletion_stats['chunks']} (was {initial_stats['chunks']})")
    
    # Add new documents
    new_docs = [
        "Artificial intelligence revolutionizes modern computing and data processing systems.",
        "Machine learning algorithms enable computers to learn patterns from large datasets.",
        "Natural language processing bridges the gap between human communication and computers."
    ]
    new_doc_ids = ["ai_doc", "ml_doc", "nlp_doc"]
    
    print(f"\nAdding {len(new_docs)} new technology-focused documents")
    retriever.add_documents_batch(new_docs, doc_ids=new_doc_ids, show_progress=False)
    
    final_stats = retriever.get_system_stats()
    print(f"Final document count: {final_stats['chunks']}")
    
    # Test search after changes
    print("\n" + "="*60)
    print("SEARCH AFTER DOCUMENT MANAGEMENT")  
    print("="*60)
    
    # Search for AI-related content (should find new docs)
    ai_query = "artificial intelligence machine learning"
    print(f"Searching for: '{ai_query}' (should find new docs)")
    
    ai_results = retriever.hybrid_search(ai_query, top_k=5)
    print("AI search results:")
    for i, (doc_id, score, metadata) in enumerate(ai_results, 1):
        text = metadata.get('text', '')[:100]
        print(f"  {i}. {doc_id}: {score:.3f} - {text}...")
    
    # Search for original Brown corpus content
    brown_query = "government political"
    print(f"\nSearching for: '{brown_query}' (original Brown content)")
    
    brown_results = retriever.hybrid_search(brown_query, top_k=5)
    print("Brown corpus search results:")
    for i, (doc_id, score, metadata) in enumerate(brown_results, 1):
        text = metadata.get('text', '')[:100]
        print(f"  {i}. {doc_id}: {score:.3f} - {text}...")
    
    # Final statistics
    print("\n" + "="*60)
    print("FINAL SYSTEM STATISTICS")
    print("="*60)
    
    for key, value in final_stats.items():
        print(f"  {key}: {value}")
    
    print(f"\nDemo complete! Managed documents: -{len(docs_to_remove)} +{len(new_docs)}")
    print(f"Net change: {final_stats['chunks'] - initial_stats['chunks']} documents")

if __name__ == "__main__":
    main()