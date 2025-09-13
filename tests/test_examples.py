"""
Test that examples run without errors
"""
import subprocess
import sys
import os
from pathlib import Path
import subprocess
import sys
import os
from pathlib import Path

def test_basic_usage():
    """Test that basic_usage.py runs successfully"""
    # Go up one level to root, then into examples
    root_dir = Path(__file__).parent.parent
    result = subprocess.run([
        sys.executable, "examples/basic_usage.py"
    ], capture_output=True, text=True, cwd=root_dir)
    
    assert result.returncode == 0, f"basic_usage.py failed: {result.stderr}"
    assert "System stats:" in result.stdout
    assert "Top results:" in result.stdout

def test_brown_corpus_demo():
    """Test that brown_corpus_demo.py runs successfully"""
    root_dir = Path(__file__).parent.parent
    result = subprocess.run([
        sys.executable, "examples/brown_corpus_demo.py"
    ], capture_output=True, text=True, cwd=root_dir)
    
    assert result.returncode == 0, f"brown_corpus_demo.py failed: {result.stderr}"
    assert "Demo complete!" in result.stdout
    assert "TESTING SEARCH" in result.stdout

def test_bm25_basic():
    """Test BM25 basic functionality"""
    # Add parent directory to path for direct imports
    sys.path.insert(0, str(Path(__file__).parent.parent))
    try:
        from bm25_chroma.bm25 import BM25
        bm25 = BM25()
        bm25.add_chunk("doc1", "machine learning algorithms")
        bm25.add_chunk("doc2", "natural language processing")
        results = bm25.search("machine learning")
        assert len(results) > 0
        assert results[0][0] == "doc1"
    finally:
        # Clean up sys.path
        sys.path.remove(str(Path(__file__).parent.parent))

def test_hybrid_retriever_basic():
    """Test HybridRetriever basic functionality"""
    sys.path.insert(0, str(Path(__file__).parent.parent))
    try:
        from bm25_chroma.hybrid_retriever import HybridRetriever
        import tempfile
        import shutil
        import time
        
        # Use temp directory
        temp_dir = tempfile.mkdtemp()
        try:
            retriever = HybridRetriever(
                chroma_path=temp_dir,
                collection_name="test_collection",
                bm25_state_path=f"{temp_dir}/test_bm25.pkl"
            )
            docs = ["machine learning", "deep learning"]
            stats = retriever.add_documents_batch(docs, show_progress=False)
            assert stats['total_documents'] == 2
            assert stats['docs_per_second'] > 0
            results = retriever.hybrid_search("learning", top_k=2)
            assert len(results) == 2
        finally:
            # Close any open connections and wait a bit for file handles to release
            try:
                if 'retriever' in locals():
                    # Try to close chroma client if it exists
                    if hasattr(retriever, 'chroma_client'):
                        # Force garbage collection on the client
                        delattr(retriever, 'chroma_client')
                time.sleep(0.1)  # Small delay to allow file handles to close
            except:
                pass
            # Try multiple times to remove the directory
            for i in range(3):
                try:
                    shutil.rmtree(temp_dir)
                    break
                except PermissionError:
                    if i < 2:  # Don't sleep on the last attempt
                        time.sleep(0.1 * (i + 1))  # Increasing delay
                    else:
                        # If all attempts fail, don't let this stop the test
                        pass
    finally:
        sys.path.remove(str(Path(__file__).parent.parent))

def test_document_deletion_with_inverted_index_consistency():
    """Test document deletion with inverted index consistency validation"""
    sys.path.insert(0, str(Path(__file__).parent.parent))
    try:
        from bm25_chroma.hybrid_retriever import HybridRetriever
        from bm25_chroma.bm25 import BM25
        import tempfile
        import shutil
        
        temp_dir = tempfile.mkdtemp()
        try:
            # Fresh isolated retriever
            retriever = HybridRetriever(
                chroma_path=temp_dir,
                collection_name="consistency_test",
                bm25_state_path=f"{temp_dir}/test_bm25.pkl"
            )
            
            # Load 10 test documents (simple, predictable content)
            docs = [f"document {i} contains word{i} and common" for i in range(10)]
            doc_ids = [f"doc_{i}" for i in range(10)]
            
            retriever.add_documents_batch(docs, doc_ids=doc_ids, show_progress=False)
            
            # Validate initial state: should have 10 documents
            initial_stats = retriever.get_system_stats()
            assert initial_stats['chunks'] == 10
            
            # Check inverted index consistency before deletion
            bm25 = retriever.bm25
            initial_vocab_size = len(bm25.vocab)
            
            # Verify "common" appears in all 10 documents
            common_postings = bm25.inverted_index.get("common", [])
            assert len(common_postings) == 10, f"Expected 'common' in 10 docs, found {len(common_postings)}"
            
            # Delete doc_0 (contains "word0" and "common")
            retriever.remove_document("doc_0")
            
            # Verify count decreased
            after_delete_stats = retriever.get_system_stats()
            assert after_delete_stats['chunks'] == 9
            
            # Check inverted index consistency after deletion
            # "common" should now appear in 9 documents (not 10)
            common_postings_after = bm25.inverted_index.get("common", [])
            assert len(common_postings_after) == 9, f"After deletion, 'common' should be in 9 docs, found {len(common_postings_after)}"
            
            # "word0" should be completely removed from vocab and index
            assert "word0" not in bm25.vocab, "'word0' should be removed from vocabulary"
            assert "word0" not in bm25.inverted_index, "'word0' should be removed from inverted index"
            
            # Add new document
            new_doc = "document new contains wordnew and common"
            retriever.add_documents_batch([new_doc], doc_ids=["doc_new"], show_progress=False)
            
            # Verify count increased
            final_stats = retriever.get_system_stats()
            assert final_stats['chunks'] == 10
            
            # Check inverted index consistency after addition
            # "common" should be back to 10 documents
            common_postings_final = bm25.inverted_index.get("common", [])
            assert len(common_postings_final) == 10, f"After addition, 'common' should be in 10 docs, found {len(common_postings_final)}"
            
            # "wordnew" should exist in vocab and index
            assert "wordnew" in bm25.vocab, "'wordnew' should be in vocabulary"
            assert "wordnew" in bm25.inverted_index, "'wordnew' should be in inverted index"
            
            # Verify posting list structure: [(freq, chunk_idx), ...] in descending order
            for term, postings in bm25.inverted_index.items():
                assert isinstance(postings, list), f"Postings for '{term}' should be a list"
                for freq, chunk_idx in postings:
                    assert isinstance(freq, int), f"Frequency should be int, got {type(freq)}"
                    assert isinstance(chunk_idx, int), f"Chunk index should be int, got {type(chunk_idx)}"
                
                # Verify descending order by frequency
                frequencies = [freq for freq, _ in postings]
                assert frequencies == sorted(frequencies, reverse=True), f"Postings for '{term}' not sorted by frequency desc"
            
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)
    finally:
        sys.path.remove(str(Path(__file__).parent.parent))

def test_bm25_deletion():
    """Test BM25-only deletion functionality"""
    sys.path.insert(0, str(Path(__file__).parent.parent))
    try:
        from bm25_chroma.bm25 import BM25
        
        bm25 = BM25()
        bm25.add_chunk("doc1", "machine learning algorithms")
        bm25.add_chunk("doc2", "natural language processing")
        
        # Verify initial state
        assert bm25.chunk_count == 2
        
        # Remove one document
        bm25.remove_chunk("doc1")
        assert bm25.chunk_count == 1
        
        # Verify remaining document still searchable
        results = bm25.search("natural language")
        assert len(results) == 1
        assert results[0][0] == "doc2"
        
        # Remove non-existent document (should not error)
        bm25.remove_chunk("non_existent")
        assert bm25.chunk_count == 1
        
    finally:
        sys.path.remove(str(Path(__file__).parent.parent))

if __name__ == "__main__":
    test_basic_usage()
    test_brown_corpus_demo() 
    test_bm25_basic()
    test_hybrid_retriever_basic()
    test_document_deletion_with_inverted_index_consistency()
    test_bm25_deletion()
    print("All tests passed!")