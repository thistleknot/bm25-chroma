"""
Test that examples run without errors
Uses hashlib for deterministic, unique document IDs
"""
import subprocess
import sys
import os
from pathlib import Path
import hashlib

def test_basic_usage():
    """Test that basic_usage.py runs successfully"""
    # Go up one level to root, then into examples
    root_dir = Path(__file__).parent.parent
    result = subprocess.run([
        sys.executable, "examples/basic_usage.py"
    ], capture_output=True, text=True, cwd=root_dir)
    
    assert result.returncode == 0, f"basic_usage.py failed: {result.stderr}"
    assert "Initial system stats:" in result.stdout
    assert "Results before deletion:" in result.stdout

def test_bm25_basic():
    """Test BM25 basic functionality with hashlib IDs"""
    sys.path.insert(0, str(Path(__file__).parent.parent))
    try:
        from bm25_chroma.bm25 import BM25
        
        # Test documents
        docs = ["machine learning algorithms", "natural language processing"]
        doc_ids = [hashlib.sha256(doc.encode()).hexdigest() for doc in docs]
        
        bm25 = BM25()
        bm25.add_chunk(doc_ids[0], docs[0])
        bm25.add_chunk(doc_ids[1], docs[1])
        
        results = bm25.search("machine learning")
        assert len(results) > 0
        assert results[0][0] == doc_ids[0]  # Should return the ML document ID
        
        # Test ID consistency
        id1 = hashlib.sha256(docs[0].encode()).hexdigest()
        id2 = hashlib.sha256(docs[0].encode()).hexdigest()
        assert id1 == id2, "Same content should generate same ID"
        
    finally:
        # Clean up sys.path
        sys.path.remove(str(Path(__file__).parent.parent))

def test_hybrid_retriever_basic():
    """Test HybridRetriever basic functionality with hashlib IDs"""
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
            doc_ids = [hashlib.sha256(doc.encode()).hexdigest() for doc in docs]
            
            stats = retriever.add_documents_batch(docs, doc_ids=doc_ids, show_progress=False)
            assert stats['total_documents'] == 2
            assert stats['docs_per_second'] > 0
            
            results = retriever.hybrid_search("learning", top_k=2)
            assert len(results) == 2
            
            # Verify IDs are consistent
            regenerated_ids = [hashlib.sha256(doc.encode()).hexdigest() for doc in docs]
            assert doc_ids == regenerated_ids, "Document IDs should be deterministic"
            
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
    """Test document deletion with inverted index consistency validation using hashlib IDs"""
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
            doc_ids = [hashlib.sha256(doc.encode()).hexdigest() for doc in docs]
            
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
            
            # Delete doc_0 using its hashlib-generated ID
            doc_0_id = hashlib.sha256(docs[0].encode()).hexdigest()
            retriever.remove_document(doc_0_id)
            
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
            new_doc_id = hashlib.sha256(new_doc.encode()).hexdigest()
            retriever.add_documents_batch([new_doc], doc_ids=[new_doc_id], show_progress=False)
            
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
            
            # Test ID consistency throughout the process
            doc_0_id_regenerated = hashlib.sha256(docs[0].encode()).hexdigest()
            assert doc_0_id == doc_0_id_regenerated, "Document ID should be deterministic"
            
            new_doc_id_regenerated = hashlib.sha256(new_doc.encode()).hexdigest()
            assert new_doc_id == new_doc_id_regenerated, "New document ID should be deterministic"
            
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)
    finally:
        sys.path.remove(str(Path(__file__).parent.parent))

def test_bm25_deletion():
    """Test BM25-only deletion functionality with hashlib IDs"""
    sys.path.insert(0, str(Path(__file__).parent.parent))
    try:
        from bm25_chroma.bm25 import BM25
        
        docs = ["machine learning algorithms", "natural language processing"]
        doc_ids = [hashlib.sha256(doc.encode()).hexdigest() for doc in docs]
        
        bm25 = BM25()
        bm25.add_chunk(doc_ids[0], docs[0])
        bm25.add_chunk(doc_ids[1], docs[1])
        
        # Verify initial state
        assert bm25.chunk_count == 2
        
        # Remove one document using its hashlib ID
        bm25.remove_chunk(doc_ids[0])
        assert bm25.chunk_count == 1
        
        # Verify remaining document still searchable
        results = bm25.search("natural language")
        assert len(results) == 1
        assert results[0][0] == doc_ids[1]  # Should return the NLP document ID
        
        # Remove non-existent document (should not error)
        fake_id = hashlib.sha256("non existent document".encode()).hexdigest()
        bm25.remove_chunk(fake_id)
        assert bm25.chunk_count == 1
        
        # Test ID consistency
        id1 = hashlib.sha256(docs[0].encode()).hexdigest()
        id2 = hashlib.sha256(docs[0].encode()).hexdigest()
        assert id1 == id2, "Same content should always generate same ID"
        
    finally:
        sys.path.remove(str(Path(__file__).parent.parent))

def test_batch_operations_with_hashlib():
    """Test batch operations using hashlib IDs"""
    sys.path.insert(0, str(Path(__file__).parent.parent))
    try:
        from bm25_chroma.hybrid_retriever import HybridRetriever
        import tempfile
        import shutil
        
        temp_dir = tempfile.mkdtemp()
        try:
            retriever = HybridRetriever(
                chroma_path=temp_dir,
                collection_name="batch_test",
                bm25_state_path=f"{temp_dir}/batch_bm25.pkl"
            )
            
            # Prepare batch documents
            docs = [
                "First document about machine learning",
                "Second document about deep learning", 
                "Third document about natural language processing",
                "Fourth document about computer vision",
                "Fifth document about reinforcement learning"
            ]
            doc_ids = [hashlib.sha256(doc.encode()).hexdigest() for doc in docs]
            
            # Test batch addition
            stats = retriever.add_documents_batch(docs, doc_ids=doc_ids, show_progress=False)
            assert stats['total_documents'] == 5
            
            # Test batch deletion (remove first 3)
            docs_to_remove = doc_ids[:3]
            retriever.remove_documents_batch(docs_to_remove)
            
            final_stats = retriever.get_system_stats()
            assert final_stats['chunks'] == 2, f"Expected 2 docs after batch deletion, got {final_stats['chunks']}"
            
            # Test search works with remaining docs
            results = retriever.hybrid_search("reinforcement learning", top_k=2)
            assert len(results) > 0, "Should find remaining documents"
            
            # Verify ID consistency for remaining docs
            remaining_docs = docs[3:]  # Last 2 docs
            regenerated_ids = [hashlib.sha256(doc.encode()).hexdigest() for doc in remaining_docs]
            expected_remaining_ids = doc_ids[3:]
            assert regenerated_ids == expected_remaining_ids, "Remaining document IDs should be consistent"
            
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)
    finally:
        sys.path.remove(str(Path(__file__).parent.parent))

if __name__ == "__main__":
    test_basic_usage()
    test_bm25_basic()
    test_hybrid_retriever_basic()
    test_document_deletion_with_inverted_index_consistency()
    test_bm25_deletion()
    test_batch_operations_with_hashlib()
    print("All tests passed!")