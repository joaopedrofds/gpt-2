"""
generate.py - Gera texto com o GPT.

    python generate.py --pretrained                 # pesos oficiais do GPT-2 (124M)
    python generate.py --ckpt ckpt.pt               # modelo treinado com train_gpt2.py
    python generate.py --pretrained --prompt "Hello, I'm a language model,"
"""

import argparse

import torch
import tiktoken

from model import GPT


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained", action="store_true", help="carrega pesos do HuggingFace")
    parser.add_argument("--ckpt", type=str, default=None, help="checkpoint salvo pelo treino")
    parser.add_argument("--prompt", type=str, default="Hello, I'm a language model,")
    parser.add_argument("--num_samples", type=int, default=5)
    parser.add_argument("--max_new_tokens", type=int, default=30)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_k", type=int, default=50)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(42)

    if args.ckpt:
        # weights_only=False porque o checkpoint também guarda o GPTConfig
        ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
        model = GPT(ckpt["config"])
        model.load_state_dict(ckpt["model"])
    else:
        model = GPT.from_pretrained("gpt2")
    model.eval()   # desliga comportamentos de treino (aqui não há dropout, mas é boa prática)
    model.to(device)

    # Texto -> tokens BPE -> tensor (num_samples, T): a mesma frase repetida
    # em cada linha do batch, para gerar várias continuações em paralelo.
    enc = tiktoken.get_encoding("gpt2")
    tokens = torch.tensor(enc.encode(args.prompt), dtype=torch.long)
    idx = tokens.unsqueeze(0).repeat(args.num_samples, 1).to(device)

    out = model.generate(idx, args.max_new_tokens,
                         temperature=args.temperature, top_k=args.top_k)

    for i in range(args.num_samples):
        ids = [t for t in out[i].tolist() if t < enc.n_vocab]  # ignora tokens de padding (vocab 50304)
        print(">", enc.decode(ids))


if __name__ == "__main__":
    main()
