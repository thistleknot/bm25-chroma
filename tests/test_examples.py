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
    """Test HybridRetriever basic functionality with ChromaDB interface"""
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
            
            # Test ChromaDB interface with different ratios
            chroma_results = retriever.query(query_texts=["learning"], n_results=2, bm25_ratio=0.5)
            assert chroma_results['documents']
            assert len(chroma_results['documents'][0]) == 2
            
            chroma_results = retriever.query(query_texts=["learning"], n_results=2, bm25_ratio=1.0)
            assert chroma_results['documents']
            assert len(chroma_results['documents'][0]) == 2

            chroma_results = retriever.query(query_texts=["learning"], n_results=2, bm25_ratio=0.0)
            assert chroma_results['documents']
            assert len(chroma_results['documents'][0]) == 2

            chroma_results = retriever.query(query_texts=["learning"], n_results=2, bm25_ratio=0.25)
            assert chroma_results['documents']
            assert len(chroma_results['documents'][0]) == 2

            chroma_results = retriever.query(query_texts=["learning"], n_results=2, bm25_ratio=0.75)
            assert chroma_results['documents']
            assert len(chroma_results['documents'][0]) == 2

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
    """Test batch operations using ChromaDB interface"""
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
            
            # Test search works with remaining docs using ChromaDB interface
            chroma_results = retriever.query(
                query_texts=["reinforcement learning"], 
                n_results=2,
                include=['documents', 'metadatas']
            )
            assert chroma_results['documents'], "Should find remaining documents"
            assert len(chroma_results['documents'][0]) > 0, "Should have search results"
            
            # Verify ID consistency for remaining docs
            remaining_docs = docs[3:]  # Last 2 docs
            regenerated_ids = [hashlib.sha256(doc.encode()).hexdigest() for doc in remaining_docs]
            expected_remaining_ids = doc_ids[3:]
            assert regenerated_ids == expected_remaining_ids, "Remaining document IDs should be consistent"
            
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)
    finally:
        sys.path.remove(str(Path(__file__).parent.parent))

def test_chromadb_interface_compatibility():
    """Test ChromaDB interface compatibility with various parameters"""
    sys.path.insert(0, str(Path(__file__).parent.parent))
    try:
        from bm25_chroma.hybrid_retriever import HybridRetriever
        import tempfile
        import shutil
        
        temp_dir = tempfile.mkdtemp()
        try:
            retriever = HybridRetriever(
                chroma_path=temp_dir,
                collection_name="interface_test",
                bm25_state_path=f"{temp_dir}/interface_bm25.pkl"
            )
            
            docs = ["machine learning algorithms", "deep learning networks", "natural language processing"]
            doc_ids = [hashlib.sha256(doc.encode()).hexdigest() for doc in docs]
            
            retriever.add_documents_batch(docs, doc_ids=doc_ids, show_progress=False)
            
            # Test various ChromaDB interface scenarios
            
            # Single query string (should be converted to list)
            results = retriever.query("machine learning", n_results=2)
            assert 'documents' in results
            assert 'metadatas' in results
            assert 'distances' in results
            
            # List of queries
            results = retriever.query(["deep learning"], n_results=2)
            assert len(results['documents']) == 1  # One query
            assert len(results['documents'][0]) <= 2  # Up to 2 results
            
            # Different include parameters
            results = retriever.query(["natural language"], n_results=1, include=['documents'])
            assert 'documents' in results
            assert 'metadatas' not in results
            
            results = retriever.query(["algorithms"], n_results=1, include=['documents', 'metadatas'])
            assert 'documents' in results
            assert 'metadatas' in results
            assert 'distances' not in results
            
            # Test with ratio parameter
            results = retriever.query(["learning"], n_results=2, bm25_ratio=0.8)
            assert 'documents' in results
            
            print("ChromaDB interface compatibility tests passed")
            
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)
    finally:
        sys.path.remove(str(Path(__file__).parent.parent))



def test_reset_collection_functionality():
    """Test reset_collection method and ensure all critical features work"""
    sys.path.insert(0, str(Path(__file__).parent.parent))
    try:
        from bm25_chroma.hybrid_retriever import HybridRetriever
        import tempfile
        import shutil
        
        temp_dir = tempfile.mkdtemp()
        try:
            retriever = HybridRetriever(
                chroma_path=temp_dir,
                collection_name="reset_test",
                bm25_state_path=f"{temp_dir}/reset_bm25.pkl"
            )
            
            # Add initial documents
            initial_docs = ["machine learning algorithms", "deep learning networks"]
            initial_ids = [hashlib.sha256(doc.encode()).hexdigest() for doc in initial_docs]
            
            retriever.add_documents_batch(initial_docs, doc_ids=initial_ids, show_progress=False)
            
            # Verify documents were added
            initial_stats = retriever.get_system_stats()
            assert initial_stats['chunks'] == 2, "Should have 2 initial documents"
            
            # Test search works
            results = retriever.query(["machine learning"], n_results=1)
            assert results['documents'], "Should find documents before reset"
            assert len(results['documents'][0]) > 0, "Should have search results"
            
            # CRITICAL TEST: reset_collection method
            retriever.reset_collection()
            
            # Verify reset worked
            reset_stats = retriever.get_system_stats()
            assert reset_stats['chunks'] == 0, "Should have 0 documents after reset"
            
            # Verify BM25 was reset
            assert retriever.bm25.chunk_count == 0, "BM25 should be empty after reset"
            assert len(retriever.bm25.vocab) == 0, "BM25 vocab should be empty after reset"
            assert len(retriever.chunk_cache) == 0, "Chunk cache should be empty after reset"
            
            # Verify ChromaDB was reset
            chroma_count = len(retriever.chroma_collection.get()["ids"])
            assert chroma_count == 0, "ChromaDB should be empty after reset"
            
            # Test that we can add documents after reset
            new_docs = ["natural language processing", "computer vision systems"]
            new_ids = [hashlib.sha256(doc.encode()).hexdigest() for doc in new_docs]
            
            retriever.add_documents_batch(new_docs, doc_ids=new_ids, show_progress=False)
            
            # Verify new documents work
            final_stats = retriever.get_system_stats()
            assert final_stats['chunks'] == 2, "Should have 2 new documents after reset and re-add"
            
            # Test search works with new documents
            results = retriever.query(["natural language"], n_results=1)
            assert results['documents'], "Should find new documents after reset"
            
            print("reset_collection() functionality test passed")
            
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)
    finally:
        sys.path.remove(str(Path(__file__).parent.parent))

def test_all_critical_methods_exist():
    """Test that all critical methods exist and are callable"""
    sys.path.insert(0, str(Path(__file__).parent.parent))
    try:
        from bm25_chroma.hybrid_retriever import HybridRetriever
        import tempfile
        import shutil
        
        temp_dir = tempfile.mkdtemp()
        try:
            retriever = HybridRetriever(
                chroma_path=temp_dir,
                collection_name="methods_test",
                bm25_state_path=f"{temp_dir}/methods_bm25.pkl"
            )
            
            # Test all critical methods exist
            critical_methods = [
                'add_documents_batch',
                'remove_document', 
                'remove_documents_batch',
                'reset_collection',
                'query',  # ChromaDB interface
                'hybrid_search',  # Legacy interface
                'search_bm25',
                'search_vector',
                'get_system_stats'
            ]
            
            for method_name in critical_methods:
                assert hasattr(retriever, method_name), f"Missing critical method: {method_name}"
                method = getattr(retriever, method_name)
                assert callable(method), f"Method {method_name} is not callable"
            
            # Test all critical attributes exist
            critical_attributes = [
                'chroma_client',
                'chroma_collection', 
                'bm25',
                'chunk_cache'
            ]
            
            for attr_name in critical_attributes:
                assert hasattr(retriever, attr_name), f"Missing critical attribute: {attr_name}"
            
            print("All critical methods and attributes exist")
            
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)
    finally:
        sys.path.remove(str(Path(__file__).parent.parent))

def test_brown_corpus_example_compatibility():
    """Test that the brown corpus example pattern works"""
    sys.path.insert(0, str(Path(__file__).parent.parent))
    try:
        from bm25_chroma.hybrid_retriever import HybridRetriever
        import tempfile
        import shutil
        
        temp_dir = tempfile.mkdtemp()
        try:
            # Simulate brown_corpus_w_ratio.py usage pattern
            retriever = HybridRetriever(
                chroma_path=temp_dir,
                collection_name="brown_test"
            )
            
            # This is the exact call from brown_corpus_w_ratio.py that was failing
            retriever.reset_collection()
            
            # Add some test documents like brown corpus would
            docs = ["Machine learning helps analyze data patterns.", "Natural language processing."]
            doc_ids = [hashlib.sha256(doc.encode()).hexdigest() for doc in docs]
            
            # Test the add_new_documents_only pattern from brown corpus
            existing_ids = set(retriever.chroma_collection.get()["ids"])
            new_pairs = [(doc, doc_id) for doc, doc_id in zip(docs, doc_ids) 
                         if doc_id not in existing_ids]
            
            assert len(new_pairs) == 2, "Should have 2 new documents to add"
            
            if new_pairs:
                new_docs, new_ids = zip(*new_pairs)
                retriever.add_documents_batch(list(new_docs), doc_ids=list(new_ids), show_progress=False)
            
            # Test the various ratio searches from brown corpus
            ratios_to_test = [0.25, 0.50, 0.75]
            
            for ratio in ratios_to_test:
                results = retriever.query(
                    query_texts=["machine learning"],
                    n_results=5,
                    bm25_ratio=ratio,
                    include=['documents', 'metadatas', 'distances']
                )
                assert 'documents' in results, f"Should return documents for ratio {ratio}"
            
            # Test document removal like brown corpus does
            retriever.remove_document(doc_ids[0])
            retriever.remove_documents_batch([doc_ids[1]])
            
            final_stats = retriever.get_system_stats()
            assert final_stats['chunks'] == 0, "Should have no documents after removal"
            
            print("Brown corpus example compatibility test passed")
            
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)
    finally:
        sys.path.remove(str(Path(__file__).parent.parent))

# Add these to your main test runner
if __name__ == "__main__":
    test_basic_usage()
    test_bm25_basic()
    test_hybrid_retriever_basic()
    test_document_deletion_with_inverted_index_consistency()
    test_bm25_deletion()
    test_batch_operations_with_hashlib()
    test_chromadb_interface_compatibility()
    test_reset_collection_functionality()  # NEW
    test_all_critical_methods_exist()      # NEW
    test_brown_corpus_example_compatibility()  # NEW
    print("All tests passed!")