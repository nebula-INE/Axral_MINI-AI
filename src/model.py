"""
model.py
Decoder-only Transformer LM（GPT系）。Phase 1〜4のスケーリング（5M〜50M）に
config の d_model / n_layers / n_heads / d_ff を変えるだけで対応できる。

依存: torch のみ（Kaggle Notebookにプリインストール済み）。
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ModelOutput:
    logits: torch.Tensor


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, attn_dropout: float, resid_dropout: float, max_seq_length: int):
        super().__init__()
        assert d_model % n_heads == 0, "d_model は n_heads で割り切れる必要があります"
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.qkv_proj = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.attn_dropout = nn.Dropout(attn_dropout)
        self.resid_dropout = nn.Dropout(resid_dropout)

        # 因果マスク（未来のトークンを見せない）
        mask = torch.tril(torch.ones(max_seq_length, max_seq_length)).view(1, 1, max_seq_length, max_seq_length)
        self.register_buffer("causal_mask", mask, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        qkv = self.qkv_proj(x)
        q, k, v = qkv.split(C, dim=2)

        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)  # (B, nh, T, hd)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        att = att.masked_fill(self.causal_mask[:, :, :T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)

        y = att @ v  # (B, nh, T, hd)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.out_proj(y))
        return y


class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TransformerBlock(nn.Module):
    def __init__(self, cfg: dict):
        super().__init__()
        d_model = cfg["d_model"]
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(
            d_model=d_model,
            n_heads=cfg["n_heads"],
            attn_dropout=cfg.get("attention_dropout", 0.1),
            resid_dropout=cfg.get("residual_dropout", 0.1),
            max_seq_length=cfg["max_seq_length"],
        )
        self.ln2 = nn.LayerNorm(d_model)
        self.ff = FeedForward(d_model, cfg["d_ff"], cfg.get("dropout", 0.1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x


class TransformerLM(nn.Module):
    """config は plan の §4.2 ハイパーパラメータの `基本設定` セクションをそのまま渡す想定。"""

    def __init__(self, cfg: dict):
        super().__init__()
        self.cfg = cfg
        vocab_size = cfg["vocab_size"]
        d_model = cfg["d_model"]
        max_seq_length = cfg["max_seq_length"]

        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_length, d_model)
        self.drop = nn.Dropout(cfg.get("dropout", 0.1))

        self.blocks = nn.ModuleList([TransformerBlock(cfg) for _ in range(cfg["n_layers"])])
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

        # weight tying（embeddingとoutput headを共有し、パラメータ数を削減）
        self.head.weight = self.token_emb.weight

        self.max_seq_length = max_seq_length
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def forward(self, input_ids: torch.Tensor) -> ModelOutput:
        B, T = input_ids.shape
        assert T <= self.max_seq_length, f"seq_len={T} が max_seq_length={self.max_seq_length} を超えています"

        pos = torch.arange(0, T, device=input_ids.device).unsqueeze(0)
        x = self.token_emb(input_ids) + self.pos_emb(pos)
        x = self.drop(x)

        for block in self.blocks:
            x = block(x)

        x = self.ln_f(x)
        logits = self.head(x)
        return ModelOutput(logits=logits)

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 128,
        eos_token_id: int | None = None,
        strategy: str = "greedy",
        temperature: float = 1.0,
        top_k: int | None = None,
    ) -> torch.Tensor:
        """自己回帰生成。§4.4評価修正版で使用する（teacher-forcingのargmaxとは別物）。"""
        self.eval()
        device = input_ids.device
        generated = input_ids

        for _ in range(max_new_tokens):
            cond = generated[:, -self.max_seq_length:]
            logits = self.forward(cond).logits[:, -1, :]  # (B, vocab)

            if strategy == "greedy":
                next_token = torch.argmax(logits, dim=-1, keepdim=True)
            else:  # sampling
                logits = logits / max(temperature, 1e-5)
                if top_k is not None:
                    v, _ = torch.topk(logits, top_k)
                    logits[logits < v[:, [-1]]] = float("-inf")
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)

            generated = torch.cat([generated, next_token], dim=1)

            if eos_token_id is not None and torch.all(next_token.squeeze(-1) == eos_token_id):
                break

        return generated
