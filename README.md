# BM-25 Chroma Hybrid Retriever

A fast, memory-efficient hybrid search system combining BM25 and vector search with Reciprocal Rank Fusion (RRF).

## Features

- **BM25**: Memory-efficient with integer indices and pre-sorted postings
  - lemmatizes, lowercase, no punctuation (replaced with space)
- **Vector Search**: Semantic similarity using ChromaDB and sentence transformers  
- **Hybrid Fusion**: Industry-standard Reciprocal Rank Fusion (RRF)
- **Dual Processing Modes**: Sequential or unified batch processing
- **State Persistence**: Automatic save/load of BM25 index
- **Document Management**: Add, remove, and update documents (chunks) with inverted index consistency

## Quickstart

```python
from bm25_chroma import HybridRetriever
import hashlib

# Initialize
retriever = HybridRetriever(
    chroma_path="./my_db",
    collection_name="my_docs"
)

# Add documents with deterministic, unique IDs
documents = [
    "Machine learning helps analyze data patterns.",
    "Natural language processing understands human text.",
    "Deep learning uses neural networks for complex tasks."
]

# Content-based surrogate keys via hashlib - avoids order dependency
# Alternatively use natural keys when available
doc_ids = [hashlib.sha256(doc.encode()).hexdigest() for doc in documents]

# Add documents
retriever.add_documents_batch(
    documents,
    doc_ids=doc_ids,  # Optional: auto-generated if not provided
    mode="unified",   # or "sequential"
    show_progress=True
)

# Search
results = retriever.hybrid_search("machine learning", top_k=5, bm25_ratio=0.5)
for doc_id, score, metadata in results:
    print(f"{doc_id[:16]}...: {score:.3f} - {metadata['text'][:100]}...")

# Document management
retriever.remove_document(doc_ids[0])  # Remove single document
retriever.remove_documents_batch(doc_ids[1:3])  # Batch removal

# Add new documents
new_docs = ["Quantum computing leverages quantum mechanics."]
new_doc_ids = [hashlib.sha256(doc.encode()).hexdigest() for doc in new_docs]
retriever.add_documents_batch(new_docs, doc_ids=new_doc_ids)
```

### Why use hashlib for document IDs?

- **Deterministic**: Same content always produces the same ID
- **Unique**: SHA256 hash collisions are extremely rare
- **Content-based**: ID reflects the actual document content
- **Database-safe**: Perfect for ensuring uniqueness across systems

## Installation

```bash
pip install bm25-chroma
```

## Core Architecture

The BM25 component maintains an inverted index with the following structure:

**Data Structure:**
- **Vocabulary Set**: `set(words)` containing all unique terms
- **Inverted Index**: `dict[word] = [(frequency, document_id), ...]`
- **Posting Lists**: Tuples ordered by frequency in descending order

**Inverted Index Consistency:**
```python
# Example inverted index structure
{
    "machine": [(3, doc_1), (2, doc_5), (1, doc_3)],  # frequency descending
    "learning": [(2, doc_1), (2, doc_2), (1, doc_4)],
    "data": [(1, doc_1), (1, doc_3)]
}
```

When documents are added or removed:
1. **Addition**: Terms added to vocabulary, posting lists updated and re-sorted
2. **Removal**: Orphaned terms removed from vocabulary, posting lists cleaned
3. **Consistency**: Document statistics and averages recalculated incrementally

## Document Management

### Adding Documents

```python
# Single batch with auto-generated IDs
retriever.add_documents_batch(documents, mode="unified")

# Single batch with custom IDs
retriever.add_documents_batch(
    documents, 
    doc_ids=["custom_1", "custom_2", "custom_3"],
    mode="unified"
)

# Multiple batches for large datasets
for batch in document_batches:
    retriever.add_documents_batch(batch, mode="unified", show_progress=True)
```

### Removing Documents

```python
# Remove single document
retriever.remove_document("doc_id_1")

# Batch removal (efficient for multiple documents)
retriever.remove_documents_batch(["doc_1", "doc_2", "doc_3"])

# Check system state after removal
stats = retriever.get_system_stats()
print(f"Documents remaining: {stats['chunks']}")
```

### Search Methods

```python
# BM25 only (keyword-based)
bm25_results = retriever.search_bm25("machine learning", top_k=10)

# Vector only (semantic similarity)
vector_results = retriever.search_vector("artificial intelligence", top_k=10)

# Hybrid search with RRF fusion (recommended)
hybrid_results = retriever.hybrid_search(
    "deep learning neural networks",
    top_k=10,
    use_rrf=True,
    rrf_k=60
)
```

## Testing

Run tests to verify functionality:

```bash
pytest tests/
```

Or run directly:
```bash
python tests/test_examples.py
```

**Test Coverage:**
- Inverted index consistency validation
- Document addition/removal workflows  
- Cross-document term tracking
- Posting list ordering verification
- Vocabulary cleanup on document removal

## Examples

- `examples/basic_usage.py` - Document management workflow with custom documents

## Processing Modes

**Unified Mode** (Recommended)
- Processes both BM25 and ChromaDB together
- Usually faster for large datasets
- Better for production use

**Sequential Mode**  
- Processes ChromaDB first, then BM25
- Better for debugging and optimization
- Separate timing for each system

## Performance

The system is designed for efficiency through incremental operations:

**No Full Recalculation**: Adding or removing documents updates only affected components:
- Vocabulary set adds/removes only new/orphaned terms
- Inverted index updates only posting lists for changed terms  
- Document statistics incrementally adjust averages and counts

**Python Native Libraries**: Heavy lifting handled by optimized built-ins:
- `Counter.most_common()` provides pre-sorted frequency lists
- `heapq.merge()` efficiently combines sorted posting lists
- `set` operations for O(1) vocabulary lookups and updates
- `defaultdict(Counter)` for sparse term-document matrices

**Batch Processing**: Configurable batch sizes balance memory usage and processing speed:
- Pending additions buffer reduces index update frequency
- Automatic flush mechanism maintains data consistency
- Progress tracking for large document collections

## API

### Main Classes

- `BM25`: Fast BM25 implementation with inverted index
- `HybridRetriever`: Main hybrid search interface

### Key Methods

**Document Management:**
- `add_documents_batch()`: Add documents in batches
- `remove_document()`: Remove single document  
- `remove_documents_batch()`: Remove multiple documents

**Search Methods:**
- `search_bm25()`: BM25-only search
- `search_vector()`: Vector-only search  
- `hybrid_search()`: Combined search with RRF

**System Methods:**
- `get_system_stats()`: Performance statistics and document counts

## License

MIT