# GPT-2 (124M) do zero em PyTorch

Implementação baseada em *Let's reproduce GPT-2 (124M)* (Andrej Karpathy).

| arquivo | conteúdo |
|---|---|
| `model.py` | arquitetura: `GPTConfig`, `CausalSelfAttention`, `MLP`, `Block`, `GPT` (+ `from_pretrained`, `generate`) |
| `train_gpt2.py` | treino no Tiny Shakespeare: AdamW, warmup + cosseno, clipping, gradient accumulation, bf16 |
| `generate.py` | geração de texto com pesos oficiais ou com um checkpoint treinado |
| `GUIA_DE_ESTUDO.md` | explicação de cada componente e relação entre teoria e código |

## Como rodar

```bash
pip install -r requirements.txt

python model.py                               # contagem de parâmetros + loss inicial (~10.82)
python generate.py --pretrained               # texto com os pesos do GPT-2 da OpenAI
python train_gpt2.py --tiny --steps 200       # treino pequeno (roda em CPU)
python generate.py --ckpt ckpt.pt --prompt "ROMEO:"
python train_gpt2.py --steps 50 --compile     # GPT-2 124M completo (GPU recomendada)
```
