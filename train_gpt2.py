"""
train_gpt2.py - Treinamento do GPT-2 do zero em um texto (Tiny Shakespeare).

Exemplos de uso:
    python train_gpt2.py                  # GPT-2 124M (ideal com GPU)
    python train_gpt2.py --tiny           # modelo pequeno, roda em CPU
    python train_gpt2.py --tiny --steps 200

Este script reúne as técnicas de treino mostradas no vídeo:
  - DataLoader simples que gera pares (x, y) deslocados em 1 token
  - AdamW com weight decay seletivo
  - Learning rate com warmup linear + decaimento cosseno
  - Gradient clipping (norma global <= 1.0)
  - Gradient accumulation (simular batches grandes com pouca memória)
  - Precisão mista bf16 (autocast) e TF32 quando há GPU
  - torch.compile (opcional)
A versão distribuída (DDP, várias GPUs) e o dataset FineWeb-Edu do vídeo
foram deixados de fora para manter o código legível; ver GUIA_DE_ESTUDO.md.
"""

import os
import math
import time
import argparse
import urllib.request

import torch
import tiktoken

from model import GPT, GPTConfig

DATA_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "input.txt")


# ---------------------------------------------------------------------------
# Dados
# ---------------------------------------------------------------------------
class DataLoaderLite:
    """
    Carrega o texto inteiro, tokeniza com o BPE do GPT-2 e entrega batches
    em sequência.

    Para um trecho contíguo de B*T + 1 tokens:
        buf = [t0, t1, t2, ..., t_{BT}]
        x   = buf[:-1]  -> entrada
        y   = buf[1:]   -> alvo (o "próximo token" de cada posição)
    Assim, uma única sequência de T tokens gera T exemplos de treino:
    prever t1 dado t0; t2 dado t0..t1; ...; t_T dado t0..t_{T-1}.
    """

    def __init__(self, B: int, T: int):
        self.B, self.T = B, T
        if not os.path.exists(DATA_PATH):
            print(f"baixando Tiny Shakespeare para {DATA_PATH} ...")
            urllib.request.urlretrieve(DATA_URL, DATA_PATH)
        with open(DATA_PATH, "r", encoding="utf-8") as f:
            text = f.read()
        enc = tiktoken.get_encoding("gpt2")
        self.tokens = torch.tensor(enc.encode(text), dtype=torch.long)
        print(f"{len(self.tokens):,} tokens carregados")
        print(f"1 época = {len(self.tokens) // (B * T)} batches")
        self.pos = 0

    def next_batch(self):
        B, T = self.B, self.T
        buf = self.tokens[self.pos: self.pos + B * T + 1]
        x = buf[:-1].view(B, T)
        y = buf[1:].view(B, T)
        self.pos += B * T
        # Se o próximo batch não couber, volta ao início (nova época).
        if self.pos + B * T + 1 > len(self.tokens):
            self.pos = 0
        return x, y


# ---------------------------------------------------------------------------
# Learning rate: warmup linear + decaimento cosseno (como no GPT-3)
# ---------------------------------------------------------------------------
def get_lr(step: int, max_lr: float, min_lr: float, warmup_steps: int, max_steps: int) -> float:
    # 1) Warmup: no início os pesos são aleatórios e os gradientes são
    #    grandes e "ruidosos"; um LR alto logo de cara pode desestabilizar.
    if step < warmup_steps:
        return max_lr * (step + 1) / warmup_steps
    # 2) Depois do fim do agendamento, mantém o mínimo.
    if step > max_steps:
        return min_lr
    # 3) Entre os dois: desce suavemente de max_lr até min_lr seguindo um
    #    meio-cosseno. Passos menores no final ajudam a "assentar" no mínimo.
    ratio = (step - warmup_steps) / (max_steps - warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * ratio))   # vai de 1 até 0
    return min_lr + coeff * (max_lr - min_lr)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tiny", action="store_true",
                        help="usa um modelo pequeno (4 camadas, 256 dims) para rodar em CPU")
    parser.add_argument("--steps", type=int, default=50, help="passos de otimização")
    parser.add_argument("--compile", action="store_true", help="usa torch.compile")
    args = parser.parse_args()

    # ---------------- dispositivo ----------------
    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    device_type = "cuda" if device.startswith("cuda") else "cpu"
    print(f"dispositivo: {device}")

    torch.manual_seed(1337)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(1337)

    # ---------------- tamanho do batch ----------------
    # O GPT-3 Small usa ~0,5M tokens por passo de otimização. Como isso não
    # cabe na memória de uma vez, processamos "micro-batches" de B x T e
    # acumulamos os gradientes por grad_accum_steps antes de chamar step().
    if args.tiny:
        config = GPTConfig(block_size=256, n_layer=4, n_head=4, n_embd=256)
        B, T = 8, 128
        total_batch_size = 8 * 128 * 4        # 4 micro-batches por passo
    else:
        # vocab 50304 = múltiplo de 128 ("número bonito"): kernels de GPU
        # rodam mais rápido. Os 47 tokens extras nunca aparecem nos dados,
        # então o modelo aprende a dar probabilidade ~0 a eles.
        config = GPTConfig(vocab_size=50304)
        B, T = 16, 1024
        total_batch_size = 524288             # 2**19 ≈ 0,5M tokens
    assert total_batch_size % (B * T) == 0
    grad_accum_steps = total_batch_size // (B * T)
    print(f"tokens por passo: {total_batch_size} | grad_accum_steps: {grad_accum_steps}")

    train_loader = DataLoaderLite(B=B, T=T)

    # TF32: nas GPUs Ampere+, multiplicações de matriz em float32 passam a
    # usar internamente uma mantissa menor -> bem mais rápido, perda mínima.
    torch.set_float32_matmul_precision("high")

    # ---------------- modelo ----------------
    model = GPT(config).to(device)
    print(f"parâmetros: {sum(p.numel() for p in model.parameters()):,}")
    if args.compile:
        # Compila o grafo do modelo: funde operações (ex.: as várias contas
        # elementwise da GELU) e reduz idas e vindas à memória da GPU.
        model = torch.compile(model)

    # ---------------- otimização ----------------
    max_lr = 6e-4                 # LR do GPT-3 Small
    min_lr = max_lr * 0.1
    max_steps = args.steps
    warmup_steps = max(1, max_steps // 10)
    optimizer = model.configure_optimizers(weight_decay=0.1, learning_rate=max_lr,
                                           device_type=device_type)

    for step in range(max_steps):
        t0 = time.time()
        optimizer.zero_grad()   # gradientes acumulam por padrão no PyTorch
        loss_accum = 0.0

        for micro_step in range(grad_accum_steps):
            x, y = train_loader.next_batch()
            x, y = x.to(device), y.to(device)
            # Precisão mista: as matmuls rodam em bfloat16 (metade da memória,
            # mais rápido); operações sensíveis (softmax, loss, LayerNorm)
            # continuam em float32. bf16 tem o mesmo expoente do fp32, então
            # não precisa de "gradient scaling" como o fp16.
            with torch.autocast(device_type=device_type, dtype=torch.bfloat16,
                                enabled=(device_type == "cuda")):
                logits, loss = model(x, y)
            # A cross_entropy faz MÉDIA dentro do micro-batch. Para que a soma
            # dos gradientes acumulados seja igual à média sobre o batch
            # inteiro, dividimos cada loss por grad_accum_steps.
            loss = loss / grad_accum_steps
            loss_accum += loss.detach()
            loss.backward()

        # Gradient clipping: se a norma global do gradiente passar de 1.0,
        # reescala todos os gradientes. Protege contra um batch "azarado"
        # que geraria um gradiente gigante e um passo destrutivo.
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        lr = get_lr(step, max_lr, min_lr, warmup_steps, max_steps)
        for group in optimizer.param_groups:
            group["lr"] = lr
        optimizer.step()

        if device_type == "cuda":
            torch.cuda.synchronize()  # espera a GPU terminar para medir o tempo
        dt = time.time() - t0
        tok_per_sec = total_batch_size / dt
        print(f"step {step:4d} | loss {loss_accum.item():.6f} | lr {lr:.4e} | "
              f"norm {norm:.4f} | dt {dt * 1000:.0f}ms | tok/s {tok_per_sec:.0f}")

    # Salva um checkpoint para usar em generate.py
    raw_model = model._orig_mod if hasattr(model, "_orig_mod") else model
    torch.save({"model": raw_model.state_dict(), "config": raw_model.config}, "ckpt.pt")
    print("checkpoint salvo em ckpt.pt")


if __name__ == "__main__":
    main()
