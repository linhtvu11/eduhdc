"""
Semantic Encoder for Educational Concepts (Contribution C1).
Projects pedagogical ontology definitions into high-dimensional hypervectors
while preserving metric proximity and structural properties (Axiom 4).
Now uses pre-trained Sentence-Transformers for true semantic grounding,
replacing the hardcoded circular structure.
"""

import sys
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import torch
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer
from scipy.stats import spearmanr

src_dir = Path(__file__).resolve().parent.parent
if str(src_dir) not in sys.path:
    sys.path.append(str(src_dir))

from eduhdc.data_loader import load_junyi_graph, CurriculumGraph


class SemanticEmbeddingEncoder:
    """
    True Semantic Encoder using MiniLM embeddings projected to HDC space.
    """
    
    def __init__(self, graph: CurriculumGraph, dim: int = 10000, is_complex: bool = False, device: Optional[str] = None):
        self.graph = graph
        self.dim = dim
        self.is_complex = is_complex
        self.device = device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        
        print("Loading SentenceTransformer model...")
        # Using a fast, lightweight sentence transformer
        self.model = SentenceTransformer('all-MiniLM-L6-v2', device=self.device)
        self.emb_dim = self.model.get_sentence_embedding_dimension()
        
        # Projection matrix from embedding space to HDC space (random Gaussian projection)
        torch.manual_seed(42)
        self.projection_matrix = torch.randn(self.emb_dim, self.dim, device=self.device) / math.sqrt(self.dim)
        
        if self.is_complex:
            # Complex projection: map to phase angles
            self.projection_matrix = self.projection_matrix * math.pi
            
        self.concept_to_idx = {name: idx for idx, name in enumerate(graph.concepts)}
        self.codebook = self._encode_all_concepts()
        
    def _encode_all_concepts(self) -> torch.Tensor:
        """Encode all concepts in the graph using their names and domains."""
        descriptions = []
        for name in self.graph.concepts:
            meta = self.graph.concept_metadata.get(name, {})
            topic = meta.get("topic", name.replace("_", " ").title())
            area = meta.get("area", "mathematics")
            
            # Create a descriptive string for the transformer
            desc = f"Educational concept in {area}: {topic}"
            descriptions.append(desc)
            
        # Get dense embeddings
        print(f"Encoding {len(descriptions)} concepts via LLM embedding...")
        embeddings = self.model.encode(descriptions, convert_to_tensor=True, device=self.device)
        
        # Project to HDC space
        projected = torch.matmul(embeddings, self.projection_matrix)
        
        if self.is_complex:
            # Convert to unitary complex vectors via phase encoding
            encoded = torch.exp(1j * projected)
        else:
            # Add non-linearity (optional, helps distribute) and normalize
            encoded = torch.tanh(projected)
            encoded = F.normalize(encoded, p=2, dim=-1)
            
        return encoded
        
    def get_concept_vector(self, name: str) -> torch.Tensor:
        idx = self.concept_to_idx[name]
        return self.codebook[idx]
        
    def get_all_vectors(self) -> torch.Tensor:
        return self.codebook
        
    def compute_similarity(self, name1: str, name2: str) -> float:
        v1 = self.get_concept_vector(name1)
        v2 = self.get_concept_vector(name2)
        if self.is_complex:
            sim = (v1 * v2.conj()).sum().real.item()
            norm = (v1.abs().pow(2).sum().sqrt() * v2.abs().pow(2).sum().sqrt()).item()
            return sim / (norm + 1e-8)
        return F.cosine_similarity(v1.unsqueeze(0), v2.unsqueeze(0)).item()


def validate_semantic_preservation(encoder: SemanticEmbeddingEncoder):
    """
    Verifies that the HDC encoded vectors preserve the similarity structure
    of the original dense embeddings.
    """
    print("\n--- Semantic Proximity Preservation Check (Axiom 4) ---")
    
    concepts = encoder.graph.concepts
    n = len(concepts)
    if n < 2:
        return
        
    hdc_sims = []
    dense_sims = []
    
    # We need the original dense embeddings for comparison
    descriptions = []
    for name in concepts:
        meta = encoder.graph.concept_metadata.get(name, {})
        topic = meta.get("topic", name.replace("_", " ").title())
        area = meta.get("area", "mathematics")
        descriptions.append(f"Educational concept in {area}: {topic}")
        
    embeddings = encoder.model.encode(descriptions, convert_to_tensor=True)
    
    for i in range(n):
        for j in range(i + 1, n):
            c1, c2 = concepts[i], concepts[j]
            
            # HDC similarity
            h_sim = encoder.compute_similarity(c1, c2)
            hdc_sims.append(h_sim)
            
            # Dense LLM embedding similarity
            d_sim = F.cosine_similarity(embeddings[i].unsqueeze(0), embeddings[j].unsqueeze(0)).item()
            dense_sims.append(d_sim)
            
    # Spearman rank correlation between Dense similarities and HDC similarities
    rho, pval = spearmanr(dense_sims, hdc_sims)
    
    print(f"  Spearman correlation between LLM embeddings and HDC projection: {rho:.4f}")
    if rho > 0.5:
        print("  [PASSED] Semantic Proximity is successfully preserved in HDC space (Axiom 4)!")
    else:
        print("  [FAILED] Projection destroyed semantic relationships.")


if __name__ == "__main__":
    print("Testing True Semantic Embedding Encoder...")
    try:
        graph = load_junyi_graph()
    except FileNotFoundError:
        print("Junyi dataset not found. Run download_datasets.py first.")
        sys.exit(1)
        
    enc_real = SemanticEmbeddingEncoder(graph=graph, dim=10000, is_complex=False)
    validate_semantic_preservation(enc_real)
    
    enc_cplx = SemanticEmbeddingEncoder(graph=graph, dim=10000, is_complex=True)
    validate_semantic_preservation(enc_cplx)
