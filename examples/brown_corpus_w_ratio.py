import nltk
from nltk.corpus import brown
from nltk.tokenize.treebank import TreebankWordDetokenizer

# Download Brown corpus if not already available
nltk.download('brown', quiet=True)

import nltk
from nltk.corpus import brown
from nltk.tokenize.treebank import TreebankWordDetokenizer
import re

# Download Brown corpus if not already available
nltk.download('brown', quiet=True)

def get_brown_docs(num_docs=500):
    """Extract clean document texts from Brown corpus with proper spacing and formatting."""
    fileids = brown.fileids()[:num_docs]
    docs = []
    detokenizer = TreebankWordDetokenizer()
    
    for fileid in fileids:
        paragraphs = brown.paras(fileid)
        doc_paragraphs = []
        
        for paragraph in paragraphs:
            para_lines = []
            
            for sentence in paragraph:
                # Convert `` and '' to proper quotes
                processed_sent = []
                for token in sentence:
                    if token == '``':
                        processed_sent.append('"')
                    elif token == "''":
                        processed_sent.append('"')
                    else:
                        processed_sent.append(token)
                
                # Use TreebankWordDetokenizer for initial formatting
                detokenized_line = detokenizer.detokenize(processed_sent)
                
                # COMPREHENSIVE FIXES for TreebankWordDetokenizer issues
                
                # 1. Fix spaced punctuation: " . " → ". "
                detokenized_line = re.sub(r'\s+([.!?])\s+', r'\1 ', detokenized_line)
                
                # 2. Fix punctuation at end of quotes: " ." → "."
                detokenized_line = re.sub(r'"\s+([.!?])', r'"\1', detokenized_line)
                
                # 3. Fix quote spacing: " word " → "word"
                detokenized_line = re.sub(r'"\s+([^"]+?)\s+"', r'"\1"', detokenized_line)
                
                # 4. Fix quotes at start/end of text
                detokenized_line = re.sub(r'^\s*"\s+', r'"', detokenized_line)  # Start quotes
                detokenized_line = re.sub(r'\s+"\s*$', r'"', detokenized_line)  # End quotes
                
                # 5. Ensure space after punctuation when followed by capital letter
                detokenized_line = re.sub(r'([.!?])([A-Z])', r'\1 \2', detokenized_line)
                
                # 6. Fix spacing around other punctuation
                detokenized_line = re.sub(r'([,:;])([A-Za-z])', r'\1 \2', detokenized_line)
                
                # 7. Clean up multiple spaces
                detokenized_line = re.sub(r'\s+', ' ', detokenized_line).strip()
                
                para_lines.append(detokenized_line)
            
            # Join sentences within a paragraph
            paragraph_text = ' '.join(para_lines)
            doc_paragraphs.append(paragraph_text)
        
        # Join paragraphs with double newlines
        doc_text = '\n\n'.join(doc_paragraphs)
        docs.append(doc_text)
    
    return docs

# Generate 500 document strings
brown_docs = get_brown_docs(500)

# Display info
print(f"Generated {len(brown_docs)} document strings from Brown corpus")
print(f"First doc length: {len(brown_docs[0])} characters")
print(f"First doc preview:\n{brown_docs[0][:1000]}...")

from bm25_chroma import HybridRetriever, BM25
import hashlib

def add_new_documents_only(retriever, documents, doc_ids, mode="unified", show_progress=True):
    """Add only documents that don't already exist in the system"""
    existing_ids = set(retriever.chroma_collection.get()["ids"])
    new_pairs = [(doc, doc_id) for doc, doc_id in zip(documents, doc_ids) 
                 if doc_id not in existing_ids]
    
    if new_pairs:
        new_docs, new_ids = zip(*new_pairs)
        retriever.add_documents_batch(list(new_docs), doc_ids=list(new_ids), 
                                    mode=mode, show_progress=show_progress)
        print(f"Added {len(new_docs)} new documents")
        return len(new_docs)
    else:
        print("No new documents to add - all already exist")
        return 0

# Initialize
retriever = HybridRetriever(
    chroma_path="./my_db",
    collection_name="my_docs"
)

retriever.reset_collection()

# Add documents
documents = brown_docs

# Content-based surrogate keys via hashlib - avoids order dependency
# Alternatively use natural keys when available
DOC_IDS = [hashlib.sha256(doc.encode()).hexdigest() for doc in brown_docs]

n = 200

# Add first batch (check for duplicates)
add_new_documents_only(retriever, documents[0:n], DOC_IDS[0:n])

queries = [
    "machine learning",
    "Machine learning helps analyze data patterns.",
    "Natural language processing understands human text.",
    "Deep learning uses neural networks for complex tasks."
]

# Search
results = retriever.hybrid_search(queries[0], top_k=5)
for doc_id, score, metadata in results:
    print(f"{doc_id[:16]}...: {score:.3f} - {metadata['text'][:100]}...")

# Document management
retriever.remove_document(DOC_IDS[n-1])  # Remove single document (last one)
retriever.remove_documents_batch(DOC_IDS[n-3:n-1])  # Batch removal (2 docs)

# Search after removal
results = retriever.hybrid_search(queries[1], top_k=5)
for doc_id, score, metadata in results:
    print(f"{doc_id[:16]}...: {score:.3f} - {metadata['text'][:100]}...")

# Add new documents
new_document_content = [
    "Quantum computing leverages quantum mechanics for advanced computation.",
    "Artificial intelligence revolutionizes modern computing systems."
]
new_doc_ids = [hashlib.sha256(doc.encode()).hexdigest() for doc in new_document_content]
add_new_documents_only(retriever, new_document_content, new_doc_ids, show_progress=False)

# Search after adding new docs
results = retriever.hybrid_search(queries[2], top_k=5)
for doc_id, score, metadata in results:
    print(f"{doc_id[:16]}...: {score:.3f} - {metadata['text'][:100]}...")

# Remove the new documents
retriever.remove_documents_batch(new_doc_ids)

# Add remaining documents (check for duplicates)
add_new_documents_only(retriever, documents[n:], DOC_IDS[n:])

# Final search
results = retriever.hybrid_search(queries[3], top_k=5, bm25_ratio=0.75)
for doc_id, score, metadata in results:
    print(f"{doc_id[:16]}...: {score:.3f} - {metadata['text'][:100]}...")

results = retriever.hybrid_search(queries[3], top_k=5, bm25_ratio=0.25)
for doc_id, score, metadata in results:
    print(f"{doc_id[:16]}...: {score:.3f} - {metadata['text'][:100]}...")

results = retriever.hybrid_search(queries[3], top_k=5, bm25_ratio=0.50)
for doc_id, score, metadata in results:
    print(f"{doc_id[:16]}...: {score:.3f} - {metadata['text'][:100]}...")