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

# Keep the existing test functions but update the internal imports
def test_bm25_basic():
    """Test BM25 basic functionality"""
    # Add parent directory to path for direct imports
    sys.path.insert(0, str(Path(__file__).parent.parent))
    try:
        # Use direct import instead of relative import
        from bm25 import BM25
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
        from hybrid_retriever import HybridRetriever as HybridRetriever
        import tempfile
        import shutil
        import time
        
        # Use temp directory
        temp_dir = tempfile.mkdtemp()
        try:
            retriever = HybridRetriever(
                chroma_path=temp_dir,
                collection_name="test_collection"
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

if __name__ == "__main__":
    test_basic_usage()
    test_brown_corpus_demo() 
    test_bm25_basic()
    test_hybrid_retriever_basic()
    print("All tests passed!")