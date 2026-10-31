"""PredicateTextStudent: text encoder for relation predicates.

Migrated from RelateAnything (https://github.com/Maelic/RelateAnything).
A 6-block bidirectional transformer (~11M parameters) distilled from DINO text tower,
designed for relation phrases with antonym separation (above/below).

Key properties:
- Antonyms more separated than random pairs (cos ~0.73 vs ~0.84)
- Full CLIP BPE vocabulary (49,408 tokens) via factorized embedding
- Output: L2-normalized [N, 512] embeddings
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

CLIP_VOCAB_SIZE = 49408
CLIP_TOKENIZER_ID = "openai/clip-vit-base-patch32"

# Tokenizer file layouts (transformers version dependent)
CLIP_TOKENIZER_FILES = ("tokenizer.json", "vocab.json", "merges.txt", 
                        "tokenizer_config.json", "special_tokens_map.json")
CLIP_TOKENIZER_LAYOUTS = (("tokenizer.json",), ("vocab.json", "merges.txt"))


def _find_colocated_tokenizer(ckpt_path: str) -> Optional[str]:
    """Return directory of checkpoint if it contains tokenizer files."""
    d = Path(ckpt_path).parent
    for layout in CLIP_TOKENIZER_LAYOUTS:
        if all((d / f).exists() for f in layout):
            return str(d)
    return None


class _TransformerBlock(nn.Module):
    """Single transformer block: LayerNorm -> MultiHeadAttention -> FFN."""
    
    def __init__(self, dim: int, heads: int, ffn_dim: int, dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim),
            nn.GELU(),
            nn.Linear(ffn_dim, dim)
        )
    
    def forward(self, x: torch.Tensor, key_padding_mask: torch.Tensor) -> torch.Tensor:
        """x: [N, L, dim], key_padding_mask: [N, L] (True = padding)."""
        h = self.norm1(x)
        attn_out, _ = self.attn(h, h, h, key_padding_mask=key_padding_mask, need_weights=False)
        x = x + attn_out
        return x + self.ffn(self.norm2(x))


class PredicateTextStudent(nn.Module):
    """Lightweight text encoder for relation predicates.
    
    Architecture:
    - Factorized token embedding: [49408, 128] -> [128, 256]
    - Learned positional encoding: [32, 256]
    - 6 transformer blocks: 256-dim, 4 heads, 1024 FFN
    - Projection head: [256] -> [512]
    - Mean pooling + L2 normalization
    
    Args:
        vocab_size: Token table size (CLIP BPE vocabulary)
        d_tok: Width of factorized token embedding (None for direct embedding)
        dim: Hidden dimension
        depth: Number of transformer blocks
        heads: Number of attention heads
        ffn_dim: FFN hidden dimension
        out_dim: Output embedding dimension
        max_len: Maximum sequence length
        token_init: Optional initial token embedding weights
    """
    
    PAD_TOKEN_ID = 0
    
    def __init__(
        self,
        vocab_size: int = CLIP_VOCAB_SIZE,
        d_tok: Optional[int] = 128,
        dim: int = 256,
        depth: int = 6,
        heads: int = 4,
        ffn_dim: int = 1024,
        out_dim: int = 512,
        max_len: int = 32,
        token_init: Optional[torch.Tensor] = None
    ):
        super().__init__()
        emb_dim = d_tok if (d_tok and d_tok < dim) else dim
        
        # Store config for checkpoint saving
        self.cfg = dict(
            vocab_size=vocab_size,
            dim=dim,
            depth=depth,
            heads=heads,
            ffn_dim=ffn_dim,
            out_dim=out_dim,
            max_len=max_len,
            d_tok=(emb_dim if emb_dim < dim else None)
        )
        self.max_len = max_len
        self.out_dim = out_dim
        
        # Token embedding (factorized if d_tok < dim)
        self.token_embedding = nn.Embedding(vocab_size, emb_dim)
        self.tok_proj = nn.Linear(emb_dim, dim, bias=False) if emb_dim < dim else nn.Identity()
        
        # Learned positional encoding
        self.positional = nn.Parameter(torch.zeros(max_len, dim))
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            _TransformerBlock(dim, heads, ffn_dim) for _ in range(depth)
        ])
        
        # Output projection
        self.ln_final = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, out_dim, bias=False)
        
        # Initialize weights
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.positional, std=0.02)
        
        if token_init is not None:
            assert token_init.shape == self.token_embedding.weight.shape
            with torch.no_grad():
                self.token_embedding.weight.copy_(token_init.float())
        
        # Tokenizer (lazy loaded)
        self._tokenizer = None
        self.tokenizer_src: Optional[str] = None  # Local dir or HF id
    
    def forward(self, ids: torch.Tensor, pad_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Encode token IDs to unit vectors.
        
        Args:
            ids: [N, L] token IDs
            pad_mask: [N, L] padding mask (True = padding)
        
        Returns:
            [N, out_dim] L2-normalized embeddings
        """
        if pad_mask is None:
            pad_mask = ids == self.PAD_TOKEN_ID
        
        L = ids.shape[1]
        
        # Embed tokens + positional encoding
        x = self.tok_proj(self.token_embedding(ids)) + self.positional[:L]
        
        # Transformer blocks
        for blk in self.blocks:
            x = blk(x, key_padding_mask=pad_mask)
        
        # Mean pooling (ignore padding)
        x = self.ln_final(x)
        keep = (~pad_mask).float().unsqueeze(-1)  # [N, L, 1]
        pooled = (x * keep).sum(1) / keep.sum(1).clamp(min=1.0)  # [N, dim]
        
        # Project and normalize
        return F.normalize(self.head(pooled), dim=-1)
    
    def _ensure_tokenizer(self):
        """Lazy load CLIP tokenizer."""
        if self._tokenizer is None:
            try:
                from transformers import CLIPTokenizer
                self._tokenizer = CLIPTokenizer.from_pretrained(
                    self.tokenizer_src or CLIP_TOKENIZER_ID
                )
            except ImportError:
                raise ImportError(
                    "transformers required for tokenization. Install: pip install transformers"
                )
        return self._tokenizer
    
    def tokenize(self, texts: List[str], device=None) -> tuple[torch.Tensor, torch.Tensor]:
        """Tokenize strings to (ids, pad_mask).
        
        Args:
            texts: List of strings
            device: Target device
        
        Returns:
            ids: [N, max_len] token IDs
            pad_mask: [N, max_len] padding mask (True = padding)
        """
        tok = self._ensure_tokenizer()
        enc = tok(texts, truncation=True, max_length=self.max_len)["input_ids"]
        
        N = len(texts)
        ids = torch.full((N, self.max_len), self.PAD_TOKEN_ID, dtype=torch.long)
        pad_mask = torch.ones((N, self.max_len), dtype=torch.bool)
        
        for i, seq in enumerate(enc):
            seq = seq[:self.max_len]
            ids[i, :len(seq)] = torch.tensor(seq, dtype=torch.long)
            pad_mask[i, :len(seq)] = False
        
        if device is not None:
            ids = ids.to(device)
            pad_mask = pad_mask.to(device)
        
        return ids, pad_mask
    
    @torch.no_grad()
    def encode_texts(self, texts: List[str], device=None, batch_size: int = 1024) -> torch.Tensor:
        """Encode strings to embeddings (batched).
        
        Args:
            texts: List of strings
            device: Target device (default: model device)
            batch_size: Batch size for encoding
        
        Returns:
            [N, out_dim] L2-normalized embeddings
        """
        device = device or next(self.parameters()).device
        outs = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            ids, pad_mask = self.tokenize(batch_texts, device=device)
            outs.append(self(ids, pad_mask))
        
        if not outs:
            return torch.zeros(0, self.out_dim, device=device)
        
        return torch.cat(outs, dim=0)
    
    @classmethod
    def from_checkpoint(cls, path: str, device=None) -> "PredicateTextStudent":
        """Load model from checkpoint.
        
        Args:
            path: Path to checkpoint (.pt file)
            device: Target device
        
        Returns:
            Loaded model in eval mode
        """
        ckpt = torch.load(path, map_location=device or "cpu", weights_only=False)
        
        # Extract config (ignore 'full' flag from RA checkpoints)
        cfg = {k: v for k, v in ckpt["cfg"].items() if k != "full"}
        
        # Build model and load weights
        model = cls(**cfg)
        model.load_state_dict(ckpt["state_dict"])
        model.float()
        
        # Try to find colocated tokenizer
        model.tokenizer_src = _find_colocated_tokenizer(path)
        
        if device is not None:
            model.to(device)
        
        return model.eval()
    
    def save(self, path: str) -> None:
        """Save model checkpoint."""
        torch.save({"cfg": self.cfg, "state_dict": self.state_dict()}, path)
    
    def num_parameters(self) -> int:
        """Count total parameters."""
        return sum(p.numel() for p in self.parameters())


def build_predicate_encoder(
    checkpoint: str,
    device: Optional[str] = None,
    download: bool = True
) -> PredicateTextStudent:
    """Build predicate encoder from checkpoint.
    
    Args:
        checkpoint: Path or HF repo (e.g., 'maelic/relsgg-vits16plus')
        device: Target device
        download: Whether to download from HF if not local
    
    Returns:
        PredicateTextStudent model
    """
    # Resolve checkpoint path
    if os.path.exists(checkpoint):
        ckpt_path = checkpoint
    elif download and "/" in checkpoint:
        # HF repo format: owner/repo or owner/repo/file.pt
        from huggingface_hub import hf_hub_download
        parts = checkpoint.split("/")
        if len(parts) == 2:
            # owner/repo -> download text_student.pt
            repo_id = checkpoint
            filename = "text_student.pt"
        else:
            # owner/repo/file.pt
            repo_id = "/".join(parts[:2])
            filename = "/".join(parts[2:])
        ckpt_path = hf_hub_download(repo_id=repo_id, filename=filename)
    else:
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
    
    return PredicateTextStudent.from_checkpoint(ckpt_path, device=device)
