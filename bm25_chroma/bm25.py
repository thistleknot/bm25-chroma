import math
import re
import heapq
from collections import defaultdict, Counter
from typing import List, Tuple, Dict, Set

# Lemmatization with graceful fallback
try:
    import nltk
    from nltk.stem import WordNetLemmatizer
    from nltk.corpus import wordnet
    
    # Download required NLTK data if not present
    try:
        nltk.data.find('corpora/wordnet')
        nltk.data.find('taggers/averaged_perceptron_tagger')
    except LookupError:
        nltk.download('wordnet', quiet=True)
        nltk.download('averaged_perceptron_tagger', quiet=True)
    
    LEMMATIZER = WordNetLemmatizer()
    LEMMATIZATION_AVAILABLE = True
    
    def get_wordnet_pos(treebank_tag):
        """Convert treebank POS tag to WordNet POS tag"""
        if treebank_tag.startswith('J'):
            return wordnet.ADJ
        elif treebank_tag.startswith('V'):
            return wordnet.VERB
        elif treebank_tag.startswith('N'):
            return wordnet.NOUN
        elif treebank_tag.startswith('R'):
            return wordnet.ADV
        else:
            return wordnet.NOUN  # Default to noun
            
except ImportError:
    LEMMATIZER = None
    LEMMATIZATION_AVAILABLE = False
    get_wordnet_pos = None


class BM25:
    """
    Memory-efficient BM25 implementation with enhanced text preprocessing.
    
    Features:
    - Integer chunk indices for memory efficiency
    - Pre-sorted posting lists for faster retrieval
    - Intelligent tokenization with lemmatization support
    - Preserves alphanumeric identifiers (part numbers, serial numbers)
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
        """
        Enhanced tokenizer with lemmatization support.
        
        Features:
        - Strips punctuation while preserving alphanumeric characters
        - Converts to lowercase
        - Lemmatizes words when NLTK is available
        - Preserves part numbers, serial numbers, and other identifiers
        """
        if not isinstance(text, str):
            return []
        
        # Extract alphanumeric tokens (preserves part numbers like "P123", "SN-456", etc.)
        raw_tokens = re.findall(r'\b[a-zA-Z0-9]+\b', text.lower())
        
        # Filter by length
        filtered_tokens = [t for t in raw_tokens if 2 <= len(t) <= 50]
        
        if not LEMMATIZATION_AVAILABLE:
            return filtered_tokens
        
        # Lemmatization with POS tagging for better accuracy
        try:
            # Get POS tags for better lemmatization
            pos_tags = nltk.pos_tag(filtered_tokens)
            
            lemmatized_tokens = []
            for token, pos_tag in pos_tags:
                # Convert to alphanumeric-only tokens (numbers pass through unchanged)
                if token.isdigit() or any(char.isdigit() for char in token):
                    # Preserve alphanumeric identifiers as-is
                    lemmatized_tokens.append(token)
                else:
                    # Lemmatize alphabetic words
                    wordnet_pos = get_wordnet_pos(pos_tag)
                    lemma = LEMMATIZER.lemmatize(token, pos=wordnet_pos)
                    lemmatized_tokens.append(lemma)
            
            return lemmatized_tokens
            
        except Exception:
            # Fallback to simple lemmatization without POS
            try:
                lemmatized_tokens = []
                for token in filtered_tokens:
                    if token.isdigit() or any(char.isdigit() for char in token):
                        lemmatized_tokens.append(token)
                    else:
                        lemma = LEMMATIZER.lemmatize(token)
                        lemmatized_tokens.append(lemma)
                return lemmatized_tokens
            except Exception:
                # Final fallback to original tokens
                return filtered_tokens
    
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
        """Add chunk using optimized Counter approach with enhanced tokenization"""
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
        """BM25 search with length normalization"""
        query_tokens = self._tokenize(query)
        if not query_tokens:
            return []
        
        chunk_scores = defaultdict(float)
        
        for term in query_tokens:
            if term not in self.inverted_index:
                continue
            
            idf = self._compute_idf(term)
            postings = self.inverted_index[term]
            
            for freq, chunk_idx in postings[:1000]:
                if chunk_idx >= len(self.chunk_lengths) or self.chunk_lengths[chunk_idx] == 0:
                    continue
                
                chunk_len = self.chunk_lengths[chunk_idx]
                
                # Standard BM25 calculation
                numerator = freq * (self.k1 + 1)
                denominator = freq + self.k1 * (1 - self.b + self.b * chunk_len / self.avg_chunk_length)
                bm25_score = idf * (numerator / denominator)
                
                # Apply Solr-style norm factor
                norm_factor = 1.0 / math.sqrt(chunk_len)
                final_score = bm25_score * norm_factor
                
                chunk_id = self.chunk_id_map[chunk_idx]
                chunk_scores[chunk_id] += final_score
        
        results = [(chunk_id, score) for chunk_id, score in chunk_scores.items()]
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]
    
    def _compute_idf(self, term: str) -> float:
        """BM25 IDF calculation"""
        n_t = len(self.inverted_index.get(term, []))
        if n_t == 0:
            return 0.0
        return math.log((self.chunk_count - n_t + 0.5) / (n_t + 0.5) + 1)
    
    def get_preprocessing_info(self) -> Dict:
        """Information about text preprocessing capabilities"""
        return {
            'lemmatization_available': LEMMATIZATION_AVAILABLE,
            'features': [
                'Strips punctuation while preserving alphanumeric',
                'Converts to lowercase',
                'Lemmatizes words when NLTK available',
                'Preserves part numbers and identifiers',
                'POS-aware lemmatization for better accuracy'
            ],
            'preserved_patterns': [
                'Part numbers (P123, SN456)',
                'Serial numbers with mixed alphanumeric',
                'Model numbers and codes',
                'Any alphanumeric identifier'
            ]
        }
    
    def get_stats(self) -> Dict:
        """System statistics"""
        total_postings = sum(len(postings) for postings in self.inverted_index.values())
        
        stats = {
            'chunks': self.chunk_count,
            'vocabulary_size': len(self.vocab),
            'total_postings': total_postings,
            'avg_postings_per_term': total_postings / len(self.inverted_index) if self.inverted_index else 0,
            'avg_chunk_length': self.avg_chunk_length,
            'memory_efficiency': 'optimized_integer_indices',
            'text_preprocessing': 'enhanced_with_lemmatization' if LEMMATIZATION_AVAILABLE else 'basic_normalization'
        }
        
        return stats