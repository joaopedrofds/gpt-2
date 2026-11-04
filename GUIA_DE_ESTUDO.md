# Guia de estudo: GPT-2 do zero

Material de apoio para entender a implementação (`model.py`, `train_gpt2.py`, `generate.py`) e escrever o relatório com as suas próprias palavras. A ordem segue o caminho de um token pelo modelo.

> **Como usar este guia:** leia cada seção com o arquivo correspondente aberto ao lado. No final de cada seção há uma pergunta de verificação. Se você não conseguir respondê-la sem olhar o texto, releia a parte e rode o código com `print(x.shape)` nos pontos indicados.

---

## 0. A ideia central em uma frase

O GPT-2 é uma função que recebe uma sequência de tokens e devolve, **para cada posição**, uma distribuição de probabilidade sobre qual será o **próximo** token. Todo o resto (atenção, MLP, normalização, otimização) existe para que essa previsão fique boa e para que o treino seja estável e eficiente.

Modelos de linguagem autoregressivos fatoram a probabilidade de um texto assim:

$$P(t_1, \dots, t_n) = \prod_{i} P(t_i \mid t_1, \dots, t_{i-1})$$

O GPT aprende cada termo $P(t_i \mid t_{<i})$. Para gerar texto, basta amostrar um token, acrescentá-lo à entrada e repetir.

---

## 1. Tokenização (BPE) — `tiktoken.get_encoding("gpt2")`

**O que é:** o texto vira uma lista de inteiros. O GPT-2 usa *Byte Pair Encoding*: começa com os 256 bytes possíveis e funde repetidamente os pares mais frequentes do corpus em novos tokens (50.000 fusões). Somando o token especial `<|endoftext|>`, o vocabulário tem **50.257** tokens.

**Por que não usar caracteres ou palavras?**
- Caracteres: as sequências ficam muito longas, e o contexto de 1024 posições cobriria pouco texto.
- Palavras: o vocabulário seria enorme e palavras novas não teriam representação.
- BPE fica no meio: palavras comuns viram 1 token; palavras raras são quebradas em pedaços. Como a base são bytes, qualquer texto pode ser representado.

**Exemplo:** `"Hello, I'm a language model,"` → `[15496, 11, 314, 1101, 257, 3303, 2746, 11]` (8 tokens).

**Pergunta:** por que o Tiny Shakespeare tem ~1,1 milhão de caracteres e só ~338 mil tokens?

---

## 2. Embeddings de token e de posição — `wte`, `wpe`

```python
tok_emb = self.transformer.wte(idx)   # (B, T, C)
pos_emb = self.transformer.wpe(pos)   # (T, C)
x = tok_emb + pos_emb                 # (B, T, C)
```

- **`wte` (V × C):** uma tabela de consulta. A linha `i` é o vetor de 768 números que representa o token `i`. Esses vetores são **aprendidos**: tokens usados em contextos parecidos acabam com vetores parecidos (a ideia de *word embeddings*, como no word2vec).
- **`wpe` (1024 × C):** um vetor aprendido para cada posição de 0 a 1023.

**Por que a posição é necessária?** A atenção, sozinha, trata a entrada como um **conjunto**: se você embaralhar os tokens, cada token recebe a mesma mistura ponderada (só que embaralhada). Sem a posição, "o cachorro mordeu o homem" e "o homem mordeu o cachorro" seriam indistinguíveis. Somar `pos_emb` "carimba" em cada vetor a sua posição.

**Diferença em relação ao artigo original:** o Transformer de 2017 usa senos e cossenos fixos; o GPT-2 **aprende** os embeddings de posição.

**Pergunta:** por que a soma `tok_emb + pos_emb` funciona mesmo com formas `(B, T, C)` e `(T, C)`? (Dica: *broadcasting*.)

---

## 3. Self-attention causal — `CausalSelfAttention`

Esta é a peça mais importante. É o **único** lugar onde a informação passa de um token para outro.

### 3.1 Query, Key, Value

Cada vetor `x_i` gera três vetores por multiplicação de matriz (a camada `c_attn`):

| vetor | intuição |
|---|---|
| `q_i` (query) | "o que eu estou procurando nos tokens anteriores?" |
| `k_j` (key) | "o que eu tenho a oferecer?" |
| `v_j` (value) | "se você prestar atenção em mim, é isto que eu te passo" |

A afinidade entre a posição `i` e a posição `j` é o produto escalar `q_i · k_j`. Quanto maior, mais `i` vai "absorver" de `v_j`.

$$\text{Attention}(Q,K,V) = \text{softmax}\left(\frac{QK^\top}{\sqrt{d_k}} + M\right)V$$

onde $M$ é a máscara causal (0 onde pode olhar, $-\infty$ onde não pode).

### 3.2 Exemplo numérico (1 cabeça, 3 tokens: "o gato dormiu")

Suponha que os scores $QK^\top/\sqrt{d_k}$ sejam:

|   | o | gato | dormiu |
|---|---|---|---|
| **o** | 1.0 | 0.8 | 0.1 |
| **gato** | 0.3 | 1.2 | 0.9 |
| **dormiu** | 1.0 | 2.0 | 0.5 |

Aplicando a máscara causal (o triângulo acima da diagonal vira $-\infty$):

|   | o | gato | dormiu |
|---|---|---|---|
| **o** | 1.0 | −∞ | −∞ |
| **gato** | 0.3 | 1.2 | −∞ |
| **dormiu** | 1.0 | 2.0 | 0.5 |

Softmax em cada linha ($e^{-\infty} = 0$):

|   | o | gato | dormiu |
|---|---|---|---|
| **o** | 1.000 | 0 | 0 |
| **gato** | 0.289 | 0.711 | 0 |
| **dormiu** | 0.231 | 0.629 | 0.140 |

A saída para "dormiu" é `0.231·v_o + 0.629·v_gato + 0.140·v_dormiu`: o verbo está "puxando" informação principalmente do sujeito. Cada linha soma 1, então a saída é uma **média ponderada** dos values.

**Por que a máscara?** No treino, a posição `i` tem que prever o token `i+1`. Se ela pudesse olhar para `i+1`, bastaria copiar a resposta. A máscara garante que o treino corresponda ao uso real (na geração, o futuro ainda não existe). Graças a ela, uma sequência de T tokens fornece **T exemplos de treino de uma vez**.

### 3.3 Por que dividir por $\sqrt{d_k}$?

Se os componentes de `q` e `k` têm média 0 e variância 1, o produto escalar de dimensão `d_k = 64` tem variância 64 (desvio padrão 8). Com scores da ordem de ±8, o softmax fica quase *one-hot*: um token recebe ~100% do peso, e os gradientes dos outros ficam ~0. Dividir por $\sqrt{64} = 8$ traz a variância de volta para ~1 e mantém o softmax "difuso" no início do treino.

### 3.4 Multi-head

Em vez de uma atenção com C = 768, usamos 12 cabeças de 64 dimensões em paralelo. Cada cabeça pode aprender um tipo diferente de relação (ex.: uma olha o token anterior, outra procura o sujeito, outra a abertura de parênteses). No código, isso é só um *reshape*:

```
(B, T, 768) --view--> (B, T, 12, 64) --transpose--> (B, 12, T, 64)
```

A dimensão das cabeças passa a funcionar como uma dimensão de batch, e o PyTorch calcula as 12 atenções em paralelo. No final, o caminho inverso concatena as cabeças, e `c_proj` mistura os resultados.

### 3.5 FlashAttention

`F.scaled_dot_product_attention(..., is_causal=True)` faz a mesma conta da versão manual (comentada no código), mas sem criar a matriz `(T, T)` na memória da GPU. Ela processa blocos e usa um truque de softmax *online*. Faz **mais** operações, mas como acessa muito menos memória, fica bem mais rápida. A lição é que o gargalo na GPU costuma ser a **memória**, e não a aritmética.

**Pergunta:** qual é o formato do tensor `att` na versão manual, e por que ele cresce quadraticamente com T?

---

## 4. MLP — `MLP`

```
(B, T, 768) -> Linear -> (B, T, 3072) -> GELU -> Linear -> (B, T, 768)
```

- Atua em **cada token separadamente**: não há troca de informação entre posições aqui.
- Uma descrição comum: a atenção é **comunicação** (reduce) e o MLP é **computação** (map). O token reúne informação na atenção e "pensa" sobre ela no MLP.
- A expansão para 4× cria um espaço maior onde a não linearidade pode separar padrões. Sem a não linearidade, duas Linear seguidas seriam equivalentes a uma única Linear.
- **GELU** em vez de ReLU: suave e com gradiente diferente de zero para entradas negativas (evita "neurônios mortos"). Usa-se a aproximação por `tanh` porque foi ela que o GPT-2 usou no treino.

Cerca de **2/3 dos parâmetros** de cada bloco estão no MLP. Há pesquisas que interpretam essas camadas como uma "memória" chave-valor de fatos.

---

## 5. LayerNorm — `ln_1`, `ln_2`, `ln_f`

Para cada token, normaliza o vetor de 768 números para média 0 e variância 1, e depois aplica escala (`gamma`) e deslocamento (`beta`) aprendidos:

$$\text{LN}(x) = \gamma \cdot \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} + \beta$$

- Diferente do BatchNorm, normaliza **ao longo das features de um único token**, e não ao longo do batch. Portanto não depende do tamanho do batch e se comporta igual no treino e na inferência.
- Mantém as ativações em uma escala controlada, o que estabiliza os gradientes.

**Pre-norm vs post-norm:** o artigo original faz `LN(x + sublayer(x))`. O GPT-2 faz `x + sublayer(LN(x))` e acrescenta uma `ln_f` no final. Com o pre-norm, o caminho residual nunca passa por uma normalização. Isso facilita o treino de redes profundas.

---

## 6. Conexões residuais — `x = x + ...`

```python
x = x + self.attn(self.ln_1(x))
x = x + self.mlp(self.ln_2(x))
```

- Pense em `x` como um "fluxo residual" que atravessa a rede. Cada sub-camada lê esse fluxo e **soma** uma pequena correção.
- **Por que funciona (backpropagation):** a derivada de `a + b` em relação a `a` é 1. No backward, a soma copia o gradiente para os dois ramos, e por isso o gradiente da loss chega intacto até a primeira camada, mesmo com 12 (ou 48) blocos. Sem isso, o gradiente seria multiplicado camada por camada e poderia sumir (*vanishing gradient*).
- No início do treino, as sub-camadas contribuem pouco (ver a inicialização), e a rede se comporta quase como a identidade. Os blocos vão "ligando" aos poucos.

---

## 7. Saída: `ln_f`, `lm_head` e weight tying

- `lm_head` (C → V) transforma o vetor final de cada posição em **logits**: uma pontuação não normalizada para cada um dos 50.257 tokens. O softmax transforma os logits em probabilidades.
- **Weight tying:** `wte.weight` e `lm_head.weight` são o **mesmo tensor** (ambos têm forma 50257 × 768). Na entrada, a matriz mapeia token → vetor. Na saída, o produto escalar com cada linha mede "quão parecido o vetor final é com o embedding de cada token". A intuição é que tokens semanticamente próximos devem ser próximos nos dois papéis. Isso economiza ~38,6M parâmetros (~30% do modelo) e foi proposto por Press & Wolf (2017).

---

## 8. Contagem de parâmetros (GPT-2 124M)

| componente | conta | parâmetros |
|---|---|---|
| `wte` | 50257 × 768 | 38.597.376 |
| `wpe` | 1024 × 768 | 786.432 |
| `ln_1` | 2 × 768 | 1.536 |
| `attn.c_attn` | 768 × 2304 + 2304 | 1.771.776 |
| `attn.c_proj` | 768 × 768 + 768 | 590.592 |
| `ln_2` | 2 × 768 | 1.536 |
| `mlp.c_fc` | 768 × 3072 + 3072 | 2.362.368 |
| `mlp.c_proj` | 3072 × 768 + 768 | 2.360.064 |
| **1 bloco** | | **7.087.872** |
| **12 blocos** | | **85.054.464** |
| `ln_f` | 2 × 768 | 1.536 |
| `lm_head` | compartilhado com `wte` | 0 |
| **Total** | | **124.439.808** |

Confira rodando `python model.py`.

---

## 9. Formas dos tensores (rastreamento completo, B=4, T=32)

| ponto do código | forma |
|---|---|
| `idx` | (4, 32) |
| `tok_emb` | (4, 32, 768) |
| `pos_emb` | (32, 768) |
| `x` após a soma | (4, 32, 768) |
| `qkv = c_attn(x)` | (4, 32, 2304) |
| `q` após view/transpose | (4, 12, 32, 64) |
| `att` (versão manual) | (4, 12, 32, 32) |
| `y = att @ v` | (4, 12, 32, 64) |
| `y` após juntar as cabeças | (4, 32, 768) |
| `c_fc(x)` no MLP | (4, 32, 3072) |
| saída de cada `Block` | (4, 32, 768) |
| `logits` | (4, 32, 50257) |
| `loss` | escalar |

Note que **a forma `(B, T, C)` se mantém igual do primeiro ao último bloco**. É isso que permite empilhar quantos blocos quisermos.

---

## 10. Função de perda — cross-entropy

```python
loss = F.cross_entropy(logits.view(-1, V), targets.view(-1))
```

- Para cada uma das B·T posições: $-\log p(\text{token correto})$. Depois, tira-se a média.
- Os alvos são a própria entrada deslocada em 1 (`x = buf[:-1]`, `y = buf[1:]`). Por isso o treino é **auto-supervisionado**: o texto é o próprio rótulo.
- **Teste de sanidade:** com pesos aleatórios, o modelo deveria atribuir ~1/50257 a cada token, logo a loss inicial ≈ $\ln(50257) \approx 10{,}82$. Se a loss inicial fosse 15 ou 20, a inicialização estaria gerando previsões confiantes e erradas.

---

## 11. Inicialização — `_init_weights`

- Pesos ~ Normal(0, 0,02) e biases = 0, como no código oficial do GPT-2.
- **Escala das camadas residuais:** `c_proj` (da atenção e do MLP) usa desvio padrão multiplicado por $1/\sqrt{2 \cdot n\_layer}$.

**Exemplo para entender:** some 100 números aleatórios com desvio padrão 1. O resultado tem desvio padrão $\sqrt{100} = 10$. O fluxo residual recebe $2 \times 12 = 24$ somas; sem a correção, as ativações cresceriam com a profundidade. Multiplicando cada contribuição por $1/\sqrt{24}$, a soma volta a ter desvio padrão ~1.

```python
x = torch.zeros(768)
for _ in range(100):
    x += torch.randn(768)              # x.std() ≈ 10
x = torch.zeros(768)
for _ in range(100):
    x += 100**-0.5 * torch.randn(768)  # x.std() ≈ 1
```

---

## 12. Otimização — `configure_optimizers` e o loop de treino

| técnica | onde | por quê |
|---|---|---|
| **AdamW** (β=(0.9, 0.95)) | `configure_optimizers` | Adam adapta o passo de cada parâmetro usando médias móveis do gradiente (momento) e do gradiente ao quadrado. O "W" separa o weight decay da atualização do gradiente. |
| **Weight decay = 0.1 só em tensores 2D** | `configure_optimizers` | Regularização: encolhe os pesos das matrizes. Biases e LayerNorm ficam de fora. |
| **Warmup + cosseno** | `get_lr` | Começa com LR baixo (os gradientes iniciais são grandes e ruidosos), sobe linearmente até 6e-4 e depois decai suavemente até 6e-5. |
| **Gradient clipping (1.0)** | loop | Limita a norma global do gradiente. Evita que um batch atípico produza um passo enorme que "quebre" o modelo. |
| **Gradient accumulation** | loop | O GPT-3 Small usa ~0,5M tokens por passo. Somamos os gradientes de vários micro-batches antes de `optimizer.step()`. **Detalhe essencial:** dividir a loss por `grad_accum_steps`, porque a cross-entropy faz média e `.backward()` soma. |
| **bf16 autocast** | loop | Matmuls em 16 bits: metade da memória e mais throughput nos Tensor Cores. O bf16 mantém o expoente do fp32, então não há overflow e não é preciso *loss scaling*. |
| **TF32** | `set_float32_matmul_precision` | Matmuls fp32 com mantissa reduzida internamente: ~8× mais rápido nas GPUs A100, com perda de precisão imperceptível. |
| **torch.compile** | `--compile` | Funde várias operações em um único kernel e reduz leituras e escritas na memória da GPU. |
| **vocab 50304** | `train_gpt2.py` | Múltiplo de 128: os kernels de GPU trabalham em blocos de potências de 2. Os 47 tokens extras nunca aparecem nos dados, e o modelo aprende a dar probabilidade ~0 a eles. |

### O ciclo de um passo de treino

1. `optimizer.zero_grad()`: o PyTorch **acumula** gradientes por padrão, então é preciso zerá-los.
2. *forward*: calcula logits e loss.
3. `loss.backward()`: a backpropagation calcula ∂loss/∂parâmetro para todos os 124M parâmetros (regra da cadeia sobre o grafo computacional).
4. clipping, definição do LR e `optimizer.step()`: atualiza os pesos.

---

## 13. Geração — `GPT.generate`

1. Roda o modelo sobre a sequência atual.
2. Pega **só a última posição**: `logits[:, -1, :]`.
3. **Temperatura:** divide os logits por τ. Com τ < 1 a distribuição fica mais "pontuda" (mais previsível); com τ > 1, mais achatada (mais criativa e com mais erros).
4. **Top-k (k = 50):** descarta todos os tokens fora dos 50 mais prováveis. Sem isso, a enorme "cauda" de tokens improváveis, somada, acaba sendo sorteada de vez em quando, e o texto desanda.
5. `torch.multinomial` sorteia um token, que é concatenado à sequência. Repete-se o processo.

Na prática, o GPT-2 é **um classificador de 50.257 classes aplicado repetidamente**.

---

## 14. Do vídeo para este código: o que foi simplificado

| no vídeo | aqui |
|---|---|
| DDP com 8 GPUs (`torchrun`) | um único dispositivo |
| FineWeb-Edu (10B tokens), shards `.npy` | Tiny Shakespeare (~338k tokens) |
| avaliação com HellaSwag e loss de validação | apenas loss de treino |
| ~19.000 passos | `--steps` (padrão 50) |

Se quiser ir além, esses são bons próximos passos e também bons tópicos para a reflexão.

---

## 15. Mapa teoria ↔ código

Adapte esta tabela aos nomes que o professor usou em aula.

| conceito teórico | onde aparece |
|---|---|
| Modelo de linguagem / probabilidade condicional | `forward` (logits → softmax) e `generate` |
| Tokenização (BPE) | `tiktoken` em `DataLoaderLite` e `generate.py` |
| Word embeddings / representações distribuídas | `wte` |
| Codificação posicional | `wpe` + soma em `forward` |
| Atenção (Q, K, V, scaled dot-product) | `CausalSelfAttention.forward` |
| Mascaramento causal / decoder-only | `is_causal=True` (ou `masked_fill` na versão manual) |
| Multi-head attention | `view` + `transpose` com `n_head` |
| Feed-forward position-wise | `MLP` |
| Funções de ativação | `nn.GELU(approximate="tanh")` |
| Normalização | `nn.LayerNorm` (pre-norm) |
| Conexões residuais | `x = x + ...` em `Block.forward` |
| Softmax e cross-entropy | `F.cross_entropy` e `F.softmax` em `generate` |
| Aprendizado auto-supervisionado | `x = buf[:-1]`, `y = buf[1:]` |
| Backpropagation / regra da cadeia | `loss.backward()` |
| Gradiente descendente / Adam | `configure_optimizers`, `optimizer.step()` |
| Regularização (weight decay) | grupos de parâmetros em `configure_optimizers` |
| Inicialização de pesos (Xavier / variância) | `_init_weights` |
| Learning-rate schedule | `get_lr` |
| Vanishing / exploding gradients | residual + LayerNorm + clipping + escala na inicialização |
| Transfer learning / modelo pré-treinado | `from_pretrained` |
| Amostragem / decodificação | `generate` (temperatura, top-k) |

---

## 16. Roteiro de experimentos (bons para o relatório)

1. `python model.py`: confira 124.439.808 parâmetros e a loss inicial ≈ 10,82.
2. `python generate.py --pretrained`: o texto deve ser coerente, o que prova que a arquitetura bate com os pesos da OpenAI.
3. `python train_gpt2.py --tiny --steps 200` e depois `python generate.py --ckpt ckpt.pt --prompt "ROMEO:"`. Observe a loss caindo e o texto ganhando "cara" de Shakespeare.
4. **Ablações** (mude uma coisa por vez e compare a loss):
   - remova `pos_emb` (o modelo perde a noção de ordem);
   - troque `is_causal=True` por `False`. A loss despenca, mas a geração vira lixo. Por quê?
   - remova as conexões residuais (`x = self.attn(...)`) com `n_layer` maior;
   - tire o `/ grad_accum_steps`;
   - teste a geração com temperatura 0,5 e 1,5.

---

## 17. Sugestão de estrutura para o relatório

1. Introdução: o que é um modelo de linguagem e o que é o GPT-2.
2. Visão geral da arquitetura (diagrama com as formas dos tensores).
3. Componentes, um por subseção: propósito, funcionamento, entradas e saídas, relação com os demais.
4. Treinamento: dados, loss, otimizador, técnicas de estabilidade e eficiência.
5. Geração de texto.
6. Relação teoria ↔ prática (a tabela da seção 15, com exemplos seus).
7. Experimentos e resultados (prints da loss, amostras de texto).
8. Reflexão individual.
9. Referências.

### Sobre a reflexão individual

Esta parte precisa ser **sua**, e ninguém pode escrevê-la por você. Algumas perguntas para guiar:
- Qual componente você achou mais difícil de entender, e o que fez a ficha cair (um print de shape, um exemplo numérico, uma parte do vídeo)?
- Teve algum erro ao rodar (shape incompatível, falta de memória, loss estranha)? Como você o resolveu?
- O que te surpreendeu? (Ex.: o modelo inteiro é "só" multiplicação de matrizes; a máscara causal transforma uma sequência em T exemplos; metade da otimização é sobre memória da GPU.)
- O que você faria diferente ou estudaria a seguir?

---

## Referências

- Karpathy, A. *Let's reproduce GPT-2 (124M)*. YouTube, 2024. https://youtu.be/l8pRSuU81PU
- Vaswani et al. *Attention Is All You Need*. 2017.
- Radford et al. *Language Models are Unsupervised Multitask Learners* (GPT-2). OpenAI, 2019.
- Brown et al. *Language Models are Few-Shot Learners* (GPT-3). 2020. (Hiperparâmetros de treino.)
- Press & Wolf. *Using the Output Embedding to Improve Language Models*. 2017. (Weight tying.)
- Dao et al. *FlashAttention*. 2022.
- Hendrycks & Gimpel. *Gaussian Error Linear Units (GELUs)*. 2016.
- Loshchilov & Hutter. *Decoupled Weight Decay Regularization* (AdamW). 2019.
