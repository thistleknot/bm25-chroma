import pickle
import time
from collections import defaultdict
from typing import List, Tuple, Dict, Set, Optional, Any
import chromadb
from chromadb.utils import embedding_functions
from tqdm import tqdm

from bm25_chroma.bm25 import BM25



def reciprocal_rank_fusion(results_list: List[List[Tuple[str, float]]], 
                          k: int = 60, top_k: int = 10) -> List[Tuple[str, float]]:
    """Industry-standard RRF for combining multiple ranked lists"""
    doc_rrf_scores = defaultdict(float)
    
    for ranked_list in results_list:
        for rank, (doc_id, _) in enumerate(ranked_list, start=1):
            doc_rrf_scores[doc_id] += 1.0 / (k + rank)
    
    sorted_docs = sorted(doc_rrf_scores.items(), key=lambda x: x[1], reverse=True)
    return sorted_docs[:top_k]


class HybridRetriever:
    """
    Hybrid ensemble combining optimized BM25 + Chroma vector search with RRF
    
    Two processing modes:
    - Sequential: Chroma first, then BM25
    - Unified: Both together (usually faster)
    """
    
    def __init__(self, 
                 chroma_path: str, 
                 collection_name: str,
                 embedding_function=None,
                 bm25_state_path: str = "optimized_bm25.pkl"):
        
        # Initialize Chroma
        self.chroma_client = chromadb.PersistentClient(path=chroma_path)
        
        if embedding_function is None:
            embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name="all-MiniLM-L6-v2"
            )
        
        try:
            self.chroma_collection = self.chroma_client.get_collection(
                name=collection_name,
                embedding_function=embedding_function
            )
        except:
            self.chroma_collection = self.chroma_client.create_collection(
                name=collection_name,
                embedding_function=embedding_function
            )
        
        # Initialize BM25
        self.bm25 = BM25()
        self.bm25_state_path = bm25_state_path
        self.chunk_cache = {}
        
        # Load existing state if available
        self._load_state()
    
    def _load_state(self):
        """Load BM25 state from disk"""
        try:
            with open(self.bm25_state_path, 'rb') as f:
                state = pickle.load(f)
                self.bm25 = state['bm25']
                self.chunk_cache = state['chunk_cache']
                print(f"Loaded BM25 state: {self.bm25.chunk_count} chunks")
        except FileNotFoundError:
            print("Starting with fresh BM25 index")
    
    def _save_state(self):
        """Save BM25 state to disk"""
        state = {
            'bm25': self.bm25,
            'chunk_cache': self.chunk_cache
        }
        with open(self.bm25_state_path, 'wb') as f:
            pickle.dump(state, f)
    
    def add_documents_batch(self, 
                           documents: List[str],
                           doc_ids: Optional[List[str]] = None,
                           mode: str = "unified",
                           chroma_batch_size: int = 32,
                           bm25_batch_size: int = 100,
                           show_progress: bool = True) -> Dict[str, Any]:
        """
        Add documents in batches with two processing modes:
        
        Mode 1 - Sequential: Process all Chroma first, then BM25
        Mode 2 - Unified: Process both together in synchronized batches
        """
        
        # Generate doc_ids if not provided
        if doc_ids is None:
            doc_ids = [f"doc_{i:06d}" for i in range(len(documents))]
        
        if len(documents) != len(doc_ids):
            raise ValueError("Documents and doc_ids must have same length")
        
        print(f"Processing {len(documents)} documents in {mode} mode")
        start_time = time.time()
        
        if mode == "sequential":
            stats = self._process_sequential(documents, doc_ids, chroma_batch_size, bm25_batch_size, show_progress)
        elif mode == "unified":
            stats = self._process_unified(documents, doc_ids, chroma_batch_size, show_progress)
        else:
            raise ValueError("Mode must be 'sequential' or 'unified'")
        
        # Save state and add timing
        self._save_state()
        total_time = time.time() - start_time
        stats['total_time_seconds'] = total_time
        stats['docs_per_second'] = len(documents) / total_time
        
        print(f"Completed in {total_time:.2f}s ({stats['docs_per_second']:.1f} docs/sec)")
        return stats
    
    def _process_sequential(self, documents, doc_ids, chroma_batch_size, bm25_batch_size, show_progress):
        """Mode 1: Sequential processing - Chroma first, then BM25"""
        
        # Phase 1: Add all to Chroma in batches
        print("Phase 1: Adding to Chroma...")
        chroma_start = time.time()
        
        if show_progress:
            chroma_pbar = tqdm(range(0, len(documents), chroma_batch_size), desc="Chroma batches")
        else:
            chroma_pbar = range(0, len(documents), chroma_batch_size)
        
        for i in chroma_pbar:
            batch_docs = documents[i:i + chroma_batch_size]
            batch_ids = doc_ids[i:i + chroma_batch_size]
            batch_metadatas = [{"document_id": doc_id} for doc_id in batch_ids]
            
            try:
                self.chroma_collection.add(
                    ids=batch_ids,
                    documents=batch_docs,
                    metadatas=batch_metadatas
                )
            except Exception as e:
                print(f"Chroma batch error at index {i}: {e}")
                continue
        
        chroma_time = time.time() - chroma_start
        
        # Phase 2: Add all to BM25 in batches  
        print("Phase 2: Adding to BM25...")
        bm25_start = time.time()
        
        if show_progress:
            bm25_pbar = tqdm(range(0, len(documents), bm25_batch_size), desc="BM25 batches")
        else:
            bm25_pbar = range(0, len(documents), bm25_batch_size)
        
        for i in bm25_pbar:
            batch_docs = documents[i:i + bm25_batch_size]
            batch_ids = doc_ids[i:i + bm25_batch_size]
            
            # Create batch for BM25
            bm25_batch = [(doc_id, doc_text) for doc_id, doc_text in zip(batch_ids, batch_docs)]
            self.bm25.add_chunks_batch(bm25_batch)
            
            # Update cache
            for doc_id, doc_text in zip(batch_ids, batch_docs):
                self.chunk_cache[doc_id] = doc_text
        
        bm25_time = time.time() - bm25_start
        
        return {
            'mode': 'sequential',
            'total_documents': len(documents),
            'chroma_time_seconds': chroma_time,
            'bm25_time_seconds': bm25_time,
            'chroma_batches': (len(documents) + chroma_batch_size - 1) // chroma_batch_size,
            'bm25_batches': (len(documents) + bm25_batch_size - 1) // bm25_batch_size,
        }
    
    def _process_unified(self, documents, doc_ids, batch_size, show_progress):
        """Mode 2: Unified processing - Both systems together"""
        
        print("Unified processing: Both systems together...")
        
        if show_progress:
            pbar = tqdm(range(0, len(documents), batch_size), desc="Unified batches")
        else:
            pbar = range(0, len(documents), batch_size)
        
        chroma_errors = 0
        bm25_errors = 0
        
        for i in pbar:
            batch_docs = documents[i:i + batch_size]
            batch_ids = doc_ids[i:i + batch_size]
            batch_metadatas = [{"document_id": doc_id} for doc_id in batch_ids]
            
            # Add to Chroma
            try:
                self.chroma_collection.add(
                    ids=batch_ids,
                    documents=batch_docs,
                    metadatas=batch_metadatas
                )
            except Exception as e:
                chroma_errors += 1
                if chroma_errors <= 3:
                    print(f"Chroma error in batch {i//batch_size}: {e}")
            
            # Add to BM25
            try:
                bm25_batch = [(doc_id, doc_text) for doc_id, doc_text in zip(batch_ids, batch_docs)]
                self.bm25.add_chunks_batch(bm25_batch)
                
                # Update cache
                for doc_id, doc_text in zip(batch_ids, batch_docs):
                    self.chunk_cache[doc_id] = doc_text
                    
            except Exception as e:
                bm25_errors += 1
                if bm25_errors <= 3:
                    print(f"BM25 error in batch {i//batch_size}: {e}")
        
        return {
            'mode': 'unified',
            'total_documents': len(documents),
            'total_batches': (len(documents) + batch_size - 1) // batch_size,
            'chroma_errors': chroma_errors,
            'bm25_errors': bm25_errors,
            'batch_size': batch_size,
        }
    
    def search_bm25(self, query: str, top_k: int = 50) -> List[Tuple[str, float]]:
        """BM25-only search"""
        return self.bm25.search(query, top_k=top_k)
    
    def search_vector(self, query: str, top_k: int = 50) -> List[Tuple[str, float]]:
        """Vector-only search"""
        try:
            results = self.chroma_collection.query(
                query_texts=[query],
                n_results=top_k,
                include=["metadatas", "distances"]
            )
            
            if not results["metadatas"] or not results["metadatas"][0]:
                return []
            
            vector_results = []
            for metadata, distance in zip(results["metadatas"][0], results["distances"][0]):
                chunk_id = metadata.get("document_id", metadata.get("id", "unknown"))
                similarity = 1.0 / (1.0 + distance)
                vector_results.append((chunk_id, similarity))
            
            return vector_results
            
        except Exception as e:
            print(f"Vector search error: {e}")
            return []
    
    def hybrid_search(self, 
                     query: str, 
                     top_k: int = 10,
                     use_rrf: bool = True,
                     rrf_k: int = 60) -> List[Tuple[str, float, Dict[str, Any]]]:
        """Hybrid ensemble search combining BM25 + Vector with RRF fusion"""
        
        # Get results from both retrievers
        bm25_results = self.search_bm25(query, top_k=50)
        vector_results = self.search_vector(query, top_k=50)
        
        if not bm25_results and not vector_results:
            return []
        
        # Combine results
        if use_rrf and bm25_results and vector_results:
            fused_results = reciprocal_rank_fusion([bm25_results, vector_results], k=rrf_k, top_k=top_k)
        else:
            # Fallback: just use BM25 or vector results
            if bm25_results:
                fused_results = bm25_results[:top_k]
            else:
                fused_results = vector_results[:top_k]
        
        # Enrich with metadata and text
        enriched_results = []
        for chunk_id, score in fused_results:
            metadata = {
                'chunk_id': chunk_id,
                'text': self.chunk_cache.get(chunk_id, ""),
                'source': 'hybrid_ensemble'
            }
            enriched_results.append((chunk_id, score, metadata))
        
        return enriched_results
    
    def get_system_stats(self) -> Dict:
        """Complete system statistics"""
        bm25_stats = self.bm25.get_stats()
        # Fix: Use "metadatas" instead of "ids" since "ids" is no longer supported
        chroma_results = self.chroma_collection.get(include=["metadatas"])
        chroma_count = len(chroma_results["ids"])  # IDs are always returned by get()
        return {
            **bm25_stats,
            'chroma_chunks': chroma_count,
            'cached_chunks': len(self.chunk_cache),
            'sync_status': 'synced' if bm25_stats['chunks'] == chroma_count else 'needs_sync'
        }
        
    def remove_document(self, doc_id: str):
        """Remove document from both BM25 and Chroma"""
        # Remove from BM25 (already implemented)
        self.bm25.remove_chunk(doc_id)
        
        # Remove from Chroma
        try:
            self.chroma_collection.delete(ids=[doc_id])
        except Exception as e:
            print(f"Chroma deletion error: {e}")
        
        # Remove from cache
        self.chunk_cache.pop(doc_id, None)
        
        # Save state
        self._save_state()

    def remove_documents_batch(self, doc_ids: List[str]):
        """Remove multiple documents efficiently"""
        # Remove from Chroma in batch
        try:
            self.chroma_collection.delete(ids=doc_ids)
        except Exception as e:
            print(f"Chroma batch deletion error: {e}")
        
        # Remove from BM25 individually (no batch method exists)
        for doc_id in doc_ids:
            self.bm25.remove_chunk(doc_id)
            self.chunk_cache.pop(doc_id, None)
        
        # Save state
        self._save_state()