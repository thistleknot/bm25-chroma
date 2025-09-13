"""
Hybrid Retriever - A fast, memory-efficient hybrid search system
"""

# Use absolute imports for better compatibility
from bm25 import BM25
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from hybrid_retriever import HybridRetriever as HybridRetriever, reciprocal_rank_fusion

__version__ = "0.1.0"
__all__ = ["BM25", "HybridRetriever", "reciprocal_rank_fusion"]