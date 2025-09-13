 
# bm25.py 
import math
import re
import heapq
from collections import defaultdict, Counter
from typing import List, Tuple, Dict, Set


class BM25:
    """
    Memory-efficient BM25 implementation with integer indices and pre-sorted postings.
    
    Features:
    - Integer chunk indices for memory efficiency
    - Pre-sorted posting lists for faster retrieval
    - Batch processing with configurable flush intervals
    - Automatic vocabulary management
    """
    
    def __init__(self, k1=1.5, b=0.75, max_postings_per_term=5000):
        self.k1 = k1
        self.b = b
        self.max_postings_per_term = max_postings_per_term
        
        # Minimal data structures
        self.vocab: Set[str] = set()                                          # Shared vocabulary
        self.inverted_index: Dict[str, List[Tuple[int, int]]] = {}           # term -> [(freq, chunk_idx), ...] SORTED
        
        # Chunk mapping (chunk_idx = Chroma's document_id from metadata)
        self.chunk_id_map: List[str] = []                                     # chunk_idx -> chunk_id 
        self.chunk_id_to_idx: Dict[str, int] = {}                            # chunk_id -> chunk_idx
        self.chunk_lengths: List[int] = []                                    # chunk_idx -> token_count
        self.chunk_texts: List[str] = []                                      # chunk_idx -> text (for removal)
        
        # Batch processing for efficiency
        self.pending_additions: Dict[str, Counter] = defaultdict(Counter)     # term -> {chunk_idx: freq}
        
        self.chunk_count = 0
        self.avg_chunk_length = 0.0
    
    def _tokenize(self, text: str) -> List[str]:
        """Fast tokenizer optimized for search"""
        tokens = re.findall(r'\b[a-zA-Z0-9]+\b', text.lower())
        return [t for t in tokens if 2 <= len(t) <= 50]
    
    def _get_or_create_chunk_idx(self, chunk_id: str) -> int:
        """Map chunk_id to integer index for memory efficiency"""
        if chunk_id in self.chunk_id_to_idx:
            return self.chunk_id_to_idx[chunk_id]
        
        chunk_idx = len(self.chunk_id_map)
        self.chunk_id_map.append(chunk_id)
        self.chunk_id_to_idx[chunk_id] = chunk_idx
        self.chunk_lengths.append(0)
        self.chunk_texts.append("")
        return chunk_idx
    
    def add_chunk(self, chunk_id: str, text: str, auto_flush=True):
        """Add chunk using optimized Counter approach"""
        if not chunk_id or not isinstance(text, str):
            raise ValueError("Invalid chunk_id or text")
        
        # Handle updates
        if chunk_id in self.chunk_id_to_idx:
            self.remove_chunk(chunk_id)
        
        tokens = self._tokenize(text)
        if not tokens:
            return
        
        chunk_idx = self._get_or_create_chunk_idx(chunk_id)
        self.chunk_lengths[chunk_idx] = len(tokens)
        self.chunk_texts[chunk_idx] = text
        
        # ✨ KEY OPTIMIZATION: Counter automatically gives us sorted frequencies!
        term_counter = Counter(tokens)
        self.vocab.update(term_counter.keys())
        
        # Buffer for batch processing
        for term, freq in term_counter.items():
            self.pending_additions[term][chunk_idx] = freq
        
        self.chunk_count += 1
        self._update_avg_chunk_length()
        
        if auto_flush:
            self._flush_pending_additions()
    
    def _flush_pending_additions(self):
        """Process pending additions with auto-sorted results"""
        for term, chunk_freq_counter in self.pending_additions.items():
            if term not in self.inverted_index:
                self.inverted_index[term] = []
            
            # Counter.most_common() gives us PRE-SORTED results!
            new_postings = [(freq, chunk_idx) for chunk_idx, freq in chunk_freq_counter.most_common()]
            
            # Merge with existing sorted postings
            existing_postings = self.inverted_index[term]
            merged_postings = self._merge_sorted_postings(existing_postings, new_postings)
            
            # Control memory usage
            if len(merged_postings) > self.max_postings_per_term:
                merged_postings = merged_postings[:self.max_postings_per_term]
            
            self.inverted_index[term] = merged_postings
        
        self.pending_additions.clear()
    
    def _merge_sorted_postings(self, existing: List[Tuple[int, int]], 
                              new: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        """Merge sorted posting lists efficiently"""
        if not existing:
            return new
        if not new:
            return existing
        
        # Use heap merge for efficiency (both lists already sorted by frequency desc)
        existing_neg = [(-freq, chunk_idx) for freq, chunk_idx in existing]
        new_neg = [(-freq, chunk_idx) for freq, chunk_idx in new]
        
        merged = [(-freq, chunk_idx) for freq, chunk_idx in heapq.merge(existing_neg, new_neg)]
        
        # Remove duplicates, keeping highest frequency
        seen_chunks = set()
        result = []
        for freq, chunk_idx in merged:
            if chunk_idx not in seen_chunks:
                result.append((freq, chunk_idx))
                seen_chunks.add(chunk_idx)
        
        return result
    
    def add_chunks_batch(self, chunks: List[Tuple[str, str]]):
        """Batch processing for maximum efficiency"""
        for chunk_id, text in chunks:
            self.add_chunk(chunk_id, text, auto_flush=False)
        self._flush_pending_additions()
    
    def remove_chunk(self, chunk_id: str):
        """Remove chunk maintaining sorted order"""
        if chunk_id not in self.chunk_id_to_idx:
            return
        
        chunk_idx = self.chunk_id_to_idx[chunk_id]
        original_text = self.chunk_texts[chunk_idx] if chunk_idx < len(self.chunk_texts) else ""
        
        if not original_text:
            return
        
        tokens = self._tokenize(original_text)
        term_counter = Counter(tokens)
        
        # Remove from inverted index
        for term in term_counter.keys():
            if term in self.inverted_index:
                self.inverted_index[term] = [
                    (freq, c_idx) for freq, c_idx in self.inverted_index[term]
                    if c_idx != chunk_idx
                ]
                if not self.inverted_index[term]:
                    del self.inverted_index[term]
                    self.vocab.discard(term)
        
        # Clean up
        self.chunk_lengths[chunk_idx] = 0
        self.chunk_texts[chunk_idx] = ""
        del self.chunk_id_to_idx[chunk_id]
        
        self.chunk_count -= 1
        self._update_avg_chunk_length()
    
    def _update_avg_chunk_length(self):
        """Update average chunk length"""
        if self.chunk_count > 0:
            total_length = sum(length for length in self.chunk_lengths if length > 0)
            self.avg_chunk_length = total_length / self.chunk_count
        else:
            self.avg_chunk_length = 0.0
    
    def search(self, query: str, top_k: int = 50) -> List[Tuple[str, float]]:
        """BM25 search with early termination thanks to pre-sorted postings"""
        query_tokens = self._tokenize(query)
        if not query_tokens:
            return []
        
        chunk_scores = defaultdict(float)
        
        for term in query_tokens:
            if term not in self.inverted_index:
                continue
            
            idf = self._compute_idf(term)
            postings = self.inverted_index[term]  # Already sorted by frequency!
            
            # Early termination for common terms
            max_postings = min(len(postings), 1000)
            
            for freq, chunk_idx in postings[:max_postings]:
                if chunk_idx >= len(self.chunk_lengths) or self.chunk_lengths[chunk_idx] == 0:
                    continue
                
                chunk_len = self.chunk_lengths[chunk_idx]
                numerator = freq * (self.k1 + 1)
                denominator = freq + self.k1 * (1 - self.b + self.b * chunk_len / self.avg_chunk_length)
                term_score = idf * (numerator / denominator)
                
                chunk_id = self.chunk_id_map[chunk_idx]
                chunk_scores[chunk_id] += term_score
        
        # Sort results
        results = [(chunk_id, score) for chunk_id, score in chunk_scores.items()]
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]
    
    def _compute_idf(self, term: str) -> float:
        """BM25 IDF calculation"""
        n_t = len(self.inverted_index.get(term, []))
        if n_t == 0:
            return 0.0
        return math.log((self.chunk_count - n_t + 0.5) / (n_t + 0.5) + 1)
    
    def get_stats(self) -> Dict:
        """System statistics"""
        total_postings = sum(len(postings) for postings in self.inverted_index.values())
        
        return {
            'chunks': self.chunk_count,
            'vocabulary_size': len(self.vocab),
            'total_postings': total_postings,
            'avg_postings_per_term': total_postings / len(self.inverted_index) if self.inverted_index else 0,
            'avg_chunk_length': self.avg_chunk_length,
            'memory_efficiency': 'optimized_integer_indices'
        } 
# ... 
 
# hybrid_retriever.py 
import pickle
import time
from collections import defaultdict
from typing import List, Tuple, Dict, Set, Optional, Any
import chromadb
from chromadb.utils import embedding_functions
from tqdm import tqdm

import pickle
import time
from collections import defaultdict
from typing import List, Tuple, Dict, Set, Optional, Any
import chromadb
from chromadb.utils import embedding_functions
from tqdm import tqdm

from bm25 import BM25


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
        print("📊 Phase 1: Adding to Chroma...")
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
                print(f"⚠️ Chroma batch error at index {i}: {e}")
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
                    print(f"⚠️ Chroma error in batch {i//batch_size}: {e}")
            
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
                    print(f"⚠️ BM25 error in batch {i//batch_size}: {e}")
        
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
            print(f"⚠️ Vector search error: {e}")
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
        chroma_count = len(self.chroma_collection.get(include=["ids"])["ids"])
        
        return {
            **bm25_stats,
            'chroma_chunks': chroma_count,
            'cached_chunks': len(self.chunk_cache),
            'sync_status': 'synced' if bm25_stats['chunks'] == chroma_count else 'needs_sync'
        }from bm25 import BM25


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
        print("📊 Phase 1: Adding to Chroma...")
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
                print(f"⚠️ Chroma batch error at index {i}: {e}")
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
                    print(f"⚠️ Chroma error in batch {i//batch_size}: {e}")
            
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
                    print(f"⚠️ BM25 error in batch {i//batch_size}: {e}")
        
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
            print(f"⚠️ Vector search error: {e}")
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
        chroma_count = len(self.chroma_collection.get(include=["ids"])["ids"])
        
        return {
            **bm25_stats,
            'chroma_chunks': chroma_count,
            'cached_chunks': len(self.chunk_cache),
            'sync_status': 'synced' if bm25_stats['chunks'] == chroma_count else 'needs_sync'
        } 
# ... 
 
# setup.py 
from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="bm25-chroma",
    version="0.1.0",
    author="Joshua Laferriere",
    author_email="laferrierejc@gmail.com",
    description="A fast, memory-efficient hybrid search system combining BM25 and vector search",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/thistleknot/bm25-chroma",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "Topic :: Text Processing :: Indexing",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0",
            "black>=22.0",
            "flake8>=4.0",
            "mypy>=0.950",
        ],
    },
) 
# ... 
 
# __init__.py 
"""
Hybrid Retriever - A fast, memory-efficient hybrid search system
"""

from .bm25 import BM25
from .hybrid_retriever import HybridRetriever, reciprocal_rank_fusion

__version__ = "0.1.0"
__all__ = ["BM25", "HybridRetriever", "reciprocal_rank_fusion"] 
# ... 
 
