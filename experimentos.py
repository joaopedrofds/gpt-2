"""
experimentos.py - Testes de "o que acontece se eu tirar essa peca?"

Treina o mesmo modelo pequeno 4 vezes, mudando UMA coisa por vez:
  1. normal        -> o modelo completo (referencia)
  2. sem_posicao   -> sem embedding de posicao (modelo nao sabe a ordem)
  3. sem_mascara   -> atencao pode olhar o futuro (trapaca)
  4. sem_residual  -> sem o "x = x + ..." (gradiente tem que passar por tudo)

Para cada um mostra:
  - loss de treino (como o modelo se sai nos dados que ele estudou)
  - loss de validacao (nos dados que ele NUNCA viu, usando atencao causal normal,
    que e como o modelo e usado de verdade para gerar texto)

Uso:  python experimentos.py [passos]     (padrao: 150)
"""

import os
import sys
import json
import torch
import torch.nn as nn
import tiktoken
from torch.nn import functional as F

import model as M
from model import GPT, GPTConfig

HERE = os.path.dirname(os.path.abspath(__file__))
STEPS = int(sys.argv[1]) if len(sys.argv) > 1 else 150
B, T = 8, 128
LR = 6e-4

# ---- dados: 90% treino / 10% validacao ----
enc = tiktoken.get_encoding("gpt2")
with open(os.path.join(HERE, "input.txt"), encoding="utf-8") as f:
    tokens = torch.tensor(enc.encode(f.read()), dtype=torch.long)
cut = int(len(tokens) * 0.9)
train_data, val_data = tokens[:cut], tokens[cut:]


def get_batch(data, gen):
    ix = torch.randint(0, len(data) - T - 1, (B,), generator=gen)
    x = torch.stack([data[i:i + T] for i in ix])
    y = torch.stack([data[i + 1:i + T + 1] for i in ix])
    return x, y


# ---- as "quebras" ----
_orig_sdpa = F.scaled_dot_product_attention
_orig_block_forward = M.Block.forward
CAUSAL_FORCED = {"on": False}  # usado so na validacao


class ZeroEmb(nn.Module):
    """Troca o embedding de posicao por zeros: o modelo nao sabe onde cada token esta."""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, pos):
        return torch.zeros(pos.size(0), self.dim)


def apply_variant(name, model):
    # sempre restaura o original antes
    M.F.scaled_dot_product_attention = _orig_sdpa
    M.Block.forward = _orig_block_forward
    if name == "sem_posicao":
        model.transformer.wpe = ZeroEmb(model.config.n_embd)
    elif name == "sem_mascara":
        def sdpa_no_mask(q, k, v, is_causal=True):
            # se estamos validando, usa a mascara de verdade
            return _orig_sdpa(q, k, v, is_causal=True if CAUSAL_FORCED["on"] else False)
        M.F.scaled_dot_product_attention = sdpa_no_mask
    elif name == "sem_residual":
        def fwd(self, x):
            x = self.attn(self.ln_1(x))   # sem "x +"
            x = self.mlp(self.ln_2(x))    # sem "x +"
            return x
        M.Block.forward = fwd


@torch.no_grad()
def validate(model, gen):
    CAUSAL_FORCED["on"] = True
    model.eval()
    losses = []
    for _ in range(20):
        x, y = get_batch(val_data, gen)
        _, loss = model(x, y)
        losses.append(loss.item())
    model.train()
    CAUSAL_FORCED["on"] = False
    return sum(losses) / len(losses)


def run(name):
    torch.manual_seed(1337)
    gen = torch.Generator().manual_seed(1337)
    cfg = GPTConfig(block_size=256, n_layer=4, n_head=4, n_embd=256)
    model = GPT(cfg)
    apply_variant(name, model)
    opt = model.configure_optimizers(0.1, LR, "cpu")
    last = []
    max_norm = 0.0
    for step in range(STEPS):
        x, y = get_batch(train_data, gen)
        opt.zero_grad()
        _, loss = model(x, y)
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0).item()
        max_norm = max(max_norm, norm)
        # warmup simples nos primeiros 10% dos passos
        lr = LR * min(1.0, (step + 1) / max(1, STEPS // 10))
        for g in opt.param_groups:
            g["lr"] = lr
        opt.step()
        last.append(loss.item())
        if step % 25 == 0 or step == STEPS - 1:
            print(f"  [{name}] passo {step:4d} loss {loss.item():.3f} norm {norm:.2f}", flush=True)
    train_loss = sum(last[-10:]) / 10
    val_loss = validate(model, gen)
    # restaura
    M.F.scaled_dot_product_attention = _orig_sdpa
    M.Block.forward = _orig_block_forward
    return dict(train=train_loss, val=val_loss, max_norm=max_norm)


if __name__ == "__main__":
    results = {}
    for name in ["normal", "sem_posicao", "sem_mascara", "sem_residual"]:
        print(f"\n=== {name} ===", flush=True)
        results[name] = run(name)
    print("\n\nRESUMO")
    print(f"{'variante':14s} {'loss treino':>12s} {'loss validacao':>15s} {'maior norm':>11s}")
    for k, r in results.items():
        print(f"{k:14s} {r['train']:12.3f} {r['val']:15.3f} {r['max_norm']:11.2f}")
    with open(os.path.join(HERE, "resultados_experimentos.json"), "w") as f:
        json.dump(results, f, indent=2)
