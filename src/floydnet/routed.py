from __future__ import annotations

import math
from typing import Callable, Optional, Sequence

import torch
import torch.nn.functional as F
from torch import nn


def _canonicalize_pivot(x: torch.Tensor, pivot_dim: int) -> torch.Tensor:
    k = x.dim() - 3
    if not 0 <= pivot_dim < k:
        raise ValueError(f"pivot_dim must be in [0, {k})")
    perm = [0, 1, 2 + pivot_dim] + [2 + i for i in range(k) if i != pivot_dim] + [2 + k]
    return x.permute(perm)


def _route_score(
    q: torch.Tensor,
    key: torch.Tensor,
    pivot_dim: int,
) -> torch.Tensor:
    k = q.dim() - 3
    q = q.movedim(2 + pivot_dim, 2 + k - 1)
    key = key.movedim(2, 2 + k - 1)
    score = (q.unsqueeze(-2) * key.unsqueeze(-3)).sum(-1)
    return score.movedim(2 + k - 1, 2 + pivot_dim)


def _select_pivots(
    q_router: torch.Tensor,
    keys: Sequence[torch.Tensor],
    pivot_dims: Sequence[int],
    route_k: Callable[[torch.Tensor], torch.Tensor],
    top_k: int,
    chunk_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    n = keys[0].shape[2 + pivot_dims[0]]
    k = q_router.dim() - 3
    top_k = min(top_k, n)
    best_scores = None
    best_idx = None

    canonical = [_canonicalize_pivot(x, r) for x, r in zip(keys, pivot_dims)]

    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        score = None
        for x in canonical:
            xr = route_k(x[:, :, start:end])
            s = _route_score(q_router, xr, 0)
            score = s if score is None else score + s
        score = score / math.sqrt(q_router.shape[-1])

        local_k = min(top_k, end - start)
        local_scores, local_pos = score.topk(local_k, dim=-1)
        local_idx = local_pos + start

        if best_scores is None:
            best_scores = local_scores
            best_idx = local_idx
        else:
            merged_scores = torch.cat((best_scores, local_scores), dim=-1)
            merged_idx = torch.cat((best_idx, local_idx), dim=-1)
            best_scores, pos = merged_scores.topk(top_k, dim=-1)
            best_idx = torch.gather(merged_idx, -1, pos)

    return best_idx, best_scores


def _gather_selected(
    x: torch.Tensor,
    pivot_dim: int,
    pivot_idx: torch.Tensor,
) -> torch.Tensor:
    k = pivot_idx.dim() - 3
    x = _canonicalize_pivot(x, pivot_dim).movedim(2, 2 + k - 1)
    x = x.unsqueeze(2 + pivot_dim)
    target_shape = pivot_idx.shape[:-1]
    x = x.expand(*target_shape, x.shape[-2], x.shape[-1])
    index = pivot_idx.unsqueeze(-1).expand(*pivot_idx.shape, x.shape[-1])
    return torch.gather(x, 2 + k, index)


def routed_pivotal_attention(
    q: torch.Tensor,
    keys: Sequence[torch.Tensor],
    values: Sequence[torch.Tensor],
    pivot_dims: Optional[Sequence[int]] = None,
    top_k: int = 8,
    route_q: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    route_k: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    pivot_chunk_size: int = 32,
    dropout: float = 0.0,
    scale: Optional[float] = None,
    training: bool = False,
    return_routing: bool = False,
    router_weight: float = 1.0,
    router_temperature: float = 1.0,
):
    if len(keys) != len(values):
        raise ValueError("keys and values must have the same length")
    if len(keys) == 0:
        raise ValueError("keys and values must be non-empty")
    k = len(keys)
    if pivot_dims is None:
        pivot_dims = list(range(k))
    if len(pivot_dims) != k:
        raise ValueError("pivot_dims must match keys")
    if top_k < 1:
        raise ValueError("top_k must be >= 1")
    if pivot_chunk_size < 1:
        raise ValueError("pivot_chunk_size must be >= 1")
    if route_q is None or route_k is None:
        raise ValueError("route_q and route_k are required")

    q_router = route_q(q)
    pivot_idx, router_scores = _select_pivots(
        q_router,
        keys,
        pivot_dims,
        route_k,
        top_k,
        pivot_chunk_size,
    )

    selected_keys = [_gather_selected(x, r, pivot_idx) for x, r in zip(keys, pivot_dims)]
    selected_values = [_gather_selected(x, r, pivot_idx) for x, r in zip(values, pivot_dims)]

    if scale is None:
        scale = 1.0 / math.sqrt(k * q.shape[-1])

    attn_scores = None
    for x in selected_keys:
        s = (q.unsqueeze(-2) * x).sum(-1)
        attn_scores = s if attn_scores is None else attn_scores + s
    attn_scores = attn_scores * scale
    if router_weight != 0.0:
        attn_scores = attn_scores + router_weight * router_scores / router_temperature
    attn = torch.softmax(attn_scores, dim=-1)
    if dropout > 0.0:
        attn = F.dropout(attn, p=dropout, training=training)

    out = None
    for x in selected_values:
        y = (attn.unsqueeze(-1) * x).sum(-2)
        out = y if out is None else out + y

    if return_routing:
        return out, pivot_idx, router_scores
    return out


class RoutedPivotalAttentionBlock(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        floyd_level: int = 2,
        top_k: int = 8,
        route_dim: Optional[int] = 16,
        pivot_chunk_size: int = 32,
        dropout: float = 0.0,
        bias: bool = False,
        router_weight: float = 1.0,
        router_temperature: float = 1.0,
    ) -> None:
        super().__init__()
        if embed_dim % num_heads:
            raise ValueError("embed_dim must be divisible by num_heads")
        if floyd_level < 1:
            raise ValueError("floyd_level must be >= 1")
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.floyd_level = floyd_level
        self.top_k = top_k
        self.pivot_chunk_size = pivot_chunk_size
        self.head_dim = embed_dim // num_heads
        self.route_dim = route_dim or self.head_dim
        self.c_qkv = nn.Linear(embed_dim, (2 * floyd_level + 1) * embed_dim, bias=bias)
        self.route_q = nn.Linear(self.head_dim, self.route_dim, bias=False)
        self.route_k = nn.Linear(self.head_dim, self.route_dim, bias=False)
        self.c_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.dropout = dropout
        self.router_weight = router_weight
        self.router_temperature = router_temperature

    def _heads(self, x: torch.Tensor) -> torch.Tensor:
        shape = x.shape[:-1]
        return x.view(*shape, self.num_heads, self.head_dim).movedim(-2, 1)

    def forward(self, x: torch.Tensor, return_routing: bool = False):
        if x.dim() != self.floyd_level + 2:
            raise ValueError(f"expected {self.floyd_level + 2}D input, got {x.dim()}D")
        qkv = self.c_qkv(x).chunk(2 * self.floyd_level + 1, dim=-1)
        q = self._heads(qkv[0])
        keys = [self._heads(t) for t in qkv[1:self.floyd_level + 1]]
        values = [self._heads(t) for t in qkv[self.floyd_level + 1:]]
        pivots = list(range(self.floyd_level))
        routed = routed_pivotal_attention(
            q,
            keys,
            values,
            pivot_dims=pivots,
            top_k=self.top_k,
            route_q=self.route_q,
            route_k=self.route_k,
            pivot_chunk_size=self.pivot_chunk_size,
            dropout=self.dropout,
            training=self.training,
            return_routing=return_routing,
            router_weight=self.router_weight,
            router_temperature=self.router_temperature,
        )
        if return_routing:
            y, idx, scores = routed
        else:
            y = routed
        y = y.movedim(1, -2).reshape_as(x)
        y = self.c_proj(y)
        y = F.dropout(y, p=self.dropout, training=self.training)
        if return_routing:
            return y, idx, scores
        return y
