"""
model.py - Implementação da arquitetura GPT-2 (124M) em PyTorch.

Baseado no vídeo "Let's reproduce GPT-2 (124M)" (Andrej Karpathy).

Visão geral do fluxo de dados (forward):

    idx (B, T)  -- índices de tokens
      │
      ├─► wte: embedding de token      -> (B, T, C)
      ├─► wpe: embedding de posição    -> (T, C)
      │        soma (broadcast)        -> (B, T, C)
      │
      ├─► N x Block:
      │       x = x + Attn(LN(x))      (comunicação entre tokens)
      │       x = x + MLP(LN(x))       (computação por token)
      │
      ├─► ln_f: LayerNorm final        -> (B, T, C)
      └─► lm_head: Linear(C -> V)      -> logits (B, T, V)

Notação usada nos comentários:
    B = batch size (nº de sequências processadas em paralelo)
    T = comprimento da sequência (nº de tokens, <= block_size)
    C = n_embd (dimensão do vetor que representa cada token)
    V = vocab_size (nº de tokens possíveis)
    nh = n_head (nº de cabeças de atenção), hs = C // nh (tamanho de cada cabeça)
"""

import math
import inspect
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F


# ---------------------------------------------------------------------------
# Configuração
# ---------------------------------------------------------------------------
@dataclass
class GPTConfig:
    # Contexto máximo: o modelo só "enxerga" até 1024 tokens anteriores.
    # Isso define o tamanho da tabela de embeddings de posição.
    block_size: int = 1024
    # 50.000 merges do BPE + 256 tokens de bytes + 1 token especial <|endoftext|>
    vocab_size: int = 50257
    n_layer: int = 12   # quantidade de blocos Transformer empilhados
    n_head: int = 12    # quantidade de cabeças de atenção por bloco
    n_embd: int = 768   # dimensão do "fluxo residual" (residual stream)


# ---------------------------------------------------------------------------
# Self-Attention causal (multi-head)
# ---------------------------------------------------------------------------
class CausalSelfAttention(nn.Module):
    """
    É aqui que os tokens "conversam" entre si.

    Cada token produz três vetores:
      - query (q): "o que eu estou procurando?"
      - key   (k): "o que eu contenho / ofereço?"
      - value (v): "o que eu passo adiante se alguém prestar atenção em mim"

    A afinidade entre o token i e o token j é q_i · k_j / sqrt(hs).
    Uma máscara causal impede que o token i olhe para j > i (o futuro),
    pois o modelo é treinado para prever o próximo token.
    """

    def __init__(self, config: GPTConfig):
        super().__init__()
        assert config.n_embd % config.n_head == 0, "n_embd precisa ser divisível por n_head"

        # Uma única camada linear produz q, k e v de uma vez (C -> 3C).
        # É matematicamente equivalente a três Linear(C, C) separadas,
        # mas uma multiplicação de matriz grande é mais eficiente na GPU.
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)

        # Projeção de saída: mistura as informações das várias cabeças
        # e devolve o resultado ao fluxo residual.
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        # Marca usada em _init_weights para escalar a inicialização
        # (ver explicação lá).
        self.c_proj.NANOGPT_SCALE_INIT = 1

        self.n_head = config.n_head
        self.n_embd = config.n_embd

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.size()

        # 1) Projeta para q, k, v: (B, T, 3C) e separa em três tensores (B, T, C)
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)

        # 2) Divide C em nh cabeças de tamanho hs e coloca a dimensão das
        #    cabeças junto do batch: (B, T, C) -> (B, T, nh, hs) -> (B, nh, T, hs).
        #    Assim cada cabeça é processada em paralelo, de forma independente,
        #    como se fosse mais uma dimensão de batch.
        hs = C // self.n_head
        q = q.view(B, T, self.n_head, hs).transpose(1, 2)
        k = k.view(B, T, self.n_head, hs).transpose(1, 2)
        v = v.view(B, T, self.n_head, hs).transpose(1, 2)

        # 3) Atenção causal. A versão "manual" equivalente seria:
        #
        #    att = (q @ k.transpose(-2, -1)) / math.sqrt(hs)     # (B, nh, T, T)
        #    mask = torch.tril(torch.ones(T, T, device=x.device)).bool()
        #    att = att.masked_fill(~mask, float('-inf'))         # esconde o futuro
        #    att = F.softmax(att, dim=-1)                        # pesos somam 1 por linha
        #    y = att @ v                                         # (B, nh, T, hs)
        #
        #    - Dividir por sqrt(hs) mantém a variância dos scores ~1; sem isso,
        #      o softmax satura (vira quase one-hot) e os gradientes somem.
        #    - -inf antes do softmax vira peso 0 depois do softmax.
        #
        #    scaled_dot_product_attention faz exatamente isso, mas usando
        #    FlashAttention quando disponível: o kernel nunca materializa a
        #    matriz (T, T) inteira na memória, o que é bem mais rápido.
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)

        # 4) Junta as cabeças de volta: (B, nh, T, hs) -> (B, T, nh, hs) -> (B, T, C)
        #    .contiguous() é necessário porque transpose só muda os "strides";
        #    view exige memória contígua.
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        # 5) Projeção de saída
        return self.c_proj(y)


# ---------------------------------------------------------------------------
# MLP (Feed-Forward)
# ---------------------------------------------------------------------------
class MLP(nn.Module):
    """
    Rede feed-forward aplicada a CADA token de forma independente.

    Se a atenção é a etapa de "comunicação" (reunir informação dos outros
    tokens), o MLP é a etapa de "computação" (processar o que foi reunido).

    Expande C -> 4C, aplica não linearidade (GELU) e projeta 4C -> C.
    """

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        # GELU com aproximação por tanh: é a versão usada no GPT-2 original
        # (na época, a GELU exata era lenta no TensorFlow). Mantemos para que
        # os pesos pré-treinados se comportem exatamente igual.
        # Diferente da ReLU, a GELU tem gradiente não nulo para x < 0,
        # evitando neurônios "mortos".
        self.gelu = nn.GELU(approximate="tanh")
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.c_fc(x)      # (B, T, 4C)
        x = self.gelu(x)      # (B, T, 4C)
        x = self.c_proj(x)    # (B, T, C)
        return x


# ---------------------------------------------------------------------------
# Bloco Transformer
# ---------------------------------------------------------------------------
class Block(nn.Module):
    """
    Um bloco do GPT-2 com "pre-norm":

        x = x + attn(ln_1(x))
        x = x + mlp(ln_2(x))

    Diferenças em relação ao Transformer original ("Attention is All You Need"):
      - Só existe o decoder (não há encoder nem cross-attention).
      - A LayerNorm vem ANTES de cada sub-camada (pre-norm), e não depois.
        Assim o caminho residual (x = x + ...) fica "limpo": o gradiente
        flui direto da saída até a entrada pela soma, o que estabiliza o
        treino de redes profundas.
    """

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Conexão residual: cada sub-camada só ADICIONA uma "correção" a x.
        # No backward, a soma distribui o gradiente igualmente para os dois
        # ramos, criando uma "autoestrada" de gradiente até as primeiras camadas.
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


# ---------------------------------------------------------------------------
# Modelo GPT completo
# ---------------------------------------------------------------------------
class GPT(nn.Module):

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config

        # Os nomes (transformer.wte, transformer.h.0.attn.c_attn, ...) seguem
        # exatamente o esquema do GPT-2 do HuggingFace, o que permite carregar
        # os pesos oficiais em from_pretrained().
        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(config.vocab_size, config.n_embd),   # token -> vetor
            wpe=nn.Embedding(config.block_size, config.n_embd),   # posição -> vetor
            h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f=nn.LayerNorm(config.n_embd),                     # LayerNorm final
        ))
        # Cabeça de linguagem: transforma o vetor final de cada posição em
        # uma pontuação (logit) para cada token do vocabulário.
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # Weight tying (compartilhamento de pesos):
        # a matriz de embedding de entrada (V, C) e a de saída (V, C) são a
        # MESMA. Intuição: tokens semanticamente parecidos devem ter
        # embeddings parecidos E probabilidades de saída parecidas.
        # Economiza 50257 * 768 ~ 38,6M parâmetros (~30% do modelo de 124M).
        self.transformer.wte.weight = self.lm_head.weight

        # Inicializa todos os pesos seguindo o GPT-2
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module):
        """
        Inicialização como no GPT-2:
          - Linear e Embedding: Normal(0, 0.02); bias = 0.
            (0.02 ~ 1/sqrt(768) ~ 0.036, ou seja, da ordem de Xavier.)
          - Camadas que escrevem no fluxo residual (c_proj) recebem o desvio
            padrão multiplicado por 1/sqrt(2 * n_layer).

        Por que o fator 1/sqrt(2 * n_layer)? Cada bloco soma DUAS
        contribuições ao fluxo residual (atenção e MLP). Somar N variáveis
        aleatórias independentes multiplica a variância por N, e o desvio
        padrão por sqrt(N). Escalando cada contribuição por 1/sqrt(N),
        o fluxo residual mantém a mesma escala no início do treino.
        (LayerNorm usa o padrão do PyTorch: gamma=1, beta=0.)
        """
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, "NANOGPT_SCALE_INIT"):
                std *= (2 * self.config.n_layer) ** -0.5
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None):
        """
        idx:     (B, T) índices de tokens
        targets: (B, T) opcional; targets[b, t] é o token correto que vem
                 depois de idx[b, :t+1] (ou seja, idx deslocado em 1).
        Retorna: logits (B, T, V) e loss (escalar) se targets for passado.
        """
        B, T = idx.size()
        assert T <= self.config.block_size, (
            f"Sequência de tamanho {T} excede block_size={self.config.block_size}"
        )

        # Embeddings de posição: o mecanismo de atenção é invariante à ordem
        # (trata a sequência como um conjunto). Somar um vetor aprendido por
        # posição é o que dá ao modelo a noção de "onde" cada token está.
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)  # (T,)
        pos_emb = self.transformer.wpe(pos)   # (T, C)
        tok_emb = self.transformer.wte(idx)   # (B, T, C)
        x = tok_emb + pos_emb                 # broadcast -> (B, T, C)

        for block in self.transformer.h:
            x = block(x)

        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)              # (B, T, V)

        loss = None
        if targets is not None:
            # Cross-entropy = -log(probabilidade atribuída ao token correto).
            # F.cross_entropy já aplica log_softmax internamente, por isso
            # passamos logits "crus". Achatamos (B, T) em uma só dimensão
            # porque cada posição é um exemplo de classificação independente.
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    # -----------------------------------------------------------------------
    # Carregar pesos oficiais do GPT-2 (HuggingFace)
    # -----------------------------------------------------------------------
    @classmethod
    def from_pretrained(cls, model_type: str = "gpt2"):
        """
        Cria o nosso GPT e copia os pesos do GPT-2 da OpenAI disponíveis
        no HuggingFace. Serve como "prova" de que a arquitetura está correta:
        se a implementação estivesse errada, os pesos não encaixariam
        ou o texto gerado seria sem sentido.
        """
        assert model_type in {"gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"}
        from transformers import GPT2LMHeadModel
        print(f"carregando pesos do modelo pré-treinado: {model_type}")

        config_args = {
            "gpt2":        dict(n_layer=12, n_head=12, n_embd=768),   # 124M
            "gpt2-medium": dict(n_layer=24, n_head=16, n_embd=1024),  # 350M
            "gpt2-large":  dict(n_layer=36, n_head=20, n_embd=1280),  # 774M
            "gpt2-xl":     dict(n_layer=48, n_head=25, n_embd=1600),  # 1558M
        }[model_type]
        config_args["vocab_size"] = 50257
        config_args["block_size"] = 1024
        model = GPT(GPTConfig(**config_args))
        sd = model.state_dict()
        # ".attn.bias" seria um buffer de máscara (não é parâmetro); ignoramos.
        sd_keys = [k for k in sd.keys() if not k.endswith(".attn.bias")]

        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()
        sd_keys_hf = [k for k in sd_hf.keys()
                      if not k.endswith(".attn.masked_bias") and not k.endswith(".attn.bias")]

        # O GPT-2 original foi escrito em TensorFlow usando "Conv1D", que
        # guarda o peso como (in, out). O nn.Linear guarda (out, in).
        # Por isso essas matrizes precisam ser transpostas.
        transposed = ["attn.c_attn.weight", "attn.c_proj.weight",
                      "mlp.c_fc.weight", "mlp.c_proj.weight"]

        assert len(sd_keys_hf) == len(sd_keys), \
            f"número de chaves diferente: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])
        return model

    # -----------------------------------------------------------------------
    # Otimizador
    # -----------------------------------------------------------------------
    def configure_optimizers(self, weight_decay: float, learning_rate: float, device_type: str):
        """
        AdamW com weight decay aplicado apenas a tensores 2D (matrizes de
        pesos das Linear e embeddings). Biases e parâmetros de LayerNorm
        (1D) não sofrem decay: são poucos parâmetros e "puxá-los" para zero
        não ajuda na regularização.

        Weight decay = regularização L2 desacoplada: a cada passo, os pesos
        encolhem um pouco, forçando o modelo a distribuir o "conhecimento"
        por muitos pesos em vez de depender de poucos pesos enormes.
        """
        params = {n: p for n, p in self.named_parameters() if p.requires_grad}
        decay_params = [p for p in params.values() if p.dim() >= 2]
        nodecay_params = [p for p in params.values() if p.dim() < 2]
        optim_groups = [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": nodecay_params, "weight_decay": 0.0},
        ]
        n_decay = sum(p.numel() for p in decay_params)
        n_nodecay = sum(p.numel() for p in nodecay_params)
        print(f"tensores com decay: {len(decay_params)} ({n_decay:,} parâmetros)")
        print(f"tensores sem decay: {len(nodecay_params)} ({n_nodecay:,} parâmetros)")

        # A versão "fused" junta as atualizações de todos os parâmetros em
        # um único kernel de GPU (menos overhead). Só existe em CUDA.
        fused_available = "fused" in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == "cuda"
        # betas=(0.9, 0.95) e eps=1e-8 seguem o artigo do GPT-3.
        return torch.optim.AdamW(optim_groups, lr=learning_rate,
                                 betas=(0.9, 0.95), eps=1e-8, fused=use_fused)

    # -----------------------------------------------------------------------
    # Geração (amostragem autoregressiva)
    # -----------------------------------------------------------------------
    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int,
                 temperature: float = 1.0, top_k: int | None = 50) -> torch.Tensor:
        """
        Gera tokens um a um. A cada passo:
          1. roda o modelo na sequência atual (cortada em block_size);
          2. pega só os logits da ÚLTIMA posição (a previsão do próximo token);
          3. divide pela temperatura (<1: mais conservador, >1: mais criativo);
          4. top-k: zera a probabilidade de tudo fora dos k mais prováveis,
             evitando que o modelo "saia do trilho" com tokens raros;
          5. amostra da distribuição e concatena o token à sequência.
        """
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.config.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature           # (B, V)
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")
            probs = F.softmax(logits, dim=-1)
            next_tok = torch.multinomial(probs, num_samples=1)  # (B, 1)
            idx = torch.cat((idx, next_tok), dim=1)
        return idx


if __name__ == "__main__":
    # Teste rápido: conta parâmetros e verifica a loss inicial.
    torch.manual_seed(1337)
    model = GPT(GPTConfig())
    n_params = sum(p.numel() for p in model.parameters())
    print(f"parâmetros: {n_params:,}")  # ~124M (weight tying conta wte uma vez só)

    x = torch.randint(0, 50257, (2, 16))
    y = torch.randint(0, 50257, (2, 16))
    logits, loss = model(x, y)
    print("logits:", tuple(logits.shape))
    # Com pesos aleatórios, o modelo deveria dar ~1/V para cada token,
    # logo a loss esperada é -ln(1/50257) ~ 10.82.
    print(f"loss inicial: {loss.item():.4f} (esperado ~ {math.log(50257):.4f})")
