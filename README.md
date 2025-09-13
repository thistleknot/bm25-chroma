# Enhanced Hybrid Retriever

A fast, memory-efficient hybrid search system combining optimized BM25 and vector search with Reciprocal Rank Fusion (RRF).

## Features

- **Optimized BM25**: Memory-efficient with integer indices and pre-sorted postings
- **Vector Search**: Semantic similarity using ChromaDB and sentence transformers  
- **Hybrid Fusion**: Industry-standard Reciprocal Rank Fusion (RRF)
- **Dual Processing Modes**: Sequential or unified batch processing
- **State Persistence**: Automatic save/load of BM25 index

## Quick Start

```python
from hybrid_retriever import EnhancedHybridRetriever

# Initialize
retriever = EnhancedHybridRetriever(
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
    mode="unified",  # or "sequential"
    show_progress=True
)

# Search
results = retriever.hybrid_search("machine learning", top_k=5)
for doc_id, score, metadata in results:
    print(f"{doc_id}: {score:.3f} - {metadata['text'][:100]}...")
```

## Installation

```bash
pip install -r requirements.txt
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

## Examples

- `examples/basic_usage.py` - Simple example with custom documents
- `examples/brown_corpus_demo.py` - Full demo with Brown corpus

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

- `BM25`: Fast BM25 implementation
- `HybridRetriever`: Main hybrid search interface

### Key Methods

- `add_documents_batch()`: Add documents in batches
- `search_bm25()`: BM25-only search
- `search_vector()`: Vector-only search  
- `hybrid_search()`: Combined search with RRF
- `get_system_stats()`: Performance statistics

## License

MIT