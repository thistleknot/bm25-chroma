"""
Complete Brown Corpus demo with mode comparison
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

def get_brown_docs(num_docs=50):
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
    print("Brown Corpus Hybrid Retriever Demo")
    
    # Get Brown corpus documents
    print("Loading Brown corpus...")
    brown_docs = get_brown_docs(50)
    print(f"Loaded {len(brown_docs)} documents")
    
    # Initialize retriever
    retriever = HybridRetriever(
        chroma_path="./brown_db",
        collection_name="brown_corpus"
    )
    
    # Test unified mode
    print("\n" + "="*50)
    print("TESTING UNIFIED MODE")
    print("="*50)
    
    unified_stats = retriever.add_documents_batch(
        brown_docs,
        doc_ids=[f"brown_{i:04d}" for i in range(len(brown_docs))],
        mode="unified",
        chroma_batch_size=8,
        show_progress=True
    )
    
    print(f"\nUnified Results:")
    print(f"  Time: {unified_stats['total_time_seconds']:.2f}s")
    print(f"  Speed: {unified_stats['docs_per_second']:.1f} docs/sec")
    
    # Test search functionality
    print("\n" + "="*50)
    print("TESTING SEARCH")
    print("="*50)
    
    test_queries = [
        "government politics",
        "economic development", 
        "social relationships"
    ]
    
    for query in test_queries:
        print(f"\nQuery: '{query}'")
        
        # BM25 search
        bm25_results = retriever.search_bm25(query, top_k=3)
        print(f"  BM25 ({len(bm25_results)} results):")
        for i, (doc_id, score) in enumerate(bm25_results[:3], 1):
            text = retriever.chunk_cache.get(doc_id, "")[:80]
            print(f"    {i}. {doc_id}: {score:.3f} - {text}...")
        
        # Vector search
        vector_results = retriever.search_vector(query, top_k=3)
        print(f"  Vector ({len(vector_results)} results):")
        for i, (doc_id, score) in enumerate(vector_results[:3], 1):
            text = retriever.chunk_cache.get(doc_id, "")[:80]
            print(f"    {i}. {doc_id}: {score:.3f} - {text}...")
        
        # Hybrid search
        hybrid_results = retriever.hybrid_search(query, top_k=3, use_rrf=True)
        print(f"  Hybrid ({len(hybrid_results)} results):")
        for i, (doc_id, score, metadata) in enumerate(hybrid_results[:3], 1):
            text = metadata.get('text', '')[:80]
            print(f"    {i}. {doc_id}: {score:.3f} - {text}...")
    
    # System statistics
    print("\n" + "="*50)
    print("FINAL STATS")
    print("="*50)
    
    stats = retriever.get_system_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    print(f"\nDemo complete! Indexed {stats['chunks']} Brown corpus documents.")


if __name__ == "__main__":
    main()