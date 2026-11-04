RELATÓRIO: IMPLEMENTAÇÃO DO GPT-2 (124M) EM PYTORCH

Aluno: [seu nome]
Disciplina: [disciplina]
Data: [data]
Vídeo de referência: Let's reproduce GPT-2 (124M), Andrej Karpathy (https://youtu.be/l8pRSuU81PU)


1. INTRODUÇÃO

O GPT-2 é um modelo de linguagem: um programa que, dado um texto, calcula qual deve ser o próximo pedaço de texto, chamado token. Para escrever frases, ele repete o processo: adivinha o próximo token, acrescenta ao texto e adivinha de novo.

Neste trabalho implementei a arquitetura do GPT-2 de 124 milhões de parâmetros, treinei uma versão pequena em um texto de Shakespeare, carreguei os pesos oficiais da OpenAI para conferir se a implementação está correta e fiz experimentos tirando peças do modelo para ver o que acontece.

Os arquivos do projeto são quatro. O model.py tem a arquitetura do modelo. O train_gpt2.py faz o treino. O generate.py gera texto. O experimentos.py roda os experimentos tirando peças do modelo.


2. VISÃO GERAL: O CAMINHO DOS DADOS

Uso estas letras para as dimensões: B é quantas sequências por vez, T é quantos tokens por sequência, C é 768 (o tamanho da "ficha" de cada token) e V é 50.257 (o tamanho do vocabulário).

O caminho é este. Entram os tokens, com formato (B, T). A tabela de tokens (wte) transforma isso em (B, T, C). A tabela de posições (wpe) dá (T, C), e a soma dos dois volta a ser (B, T, C). Depois vêm 12 blocos, e cada um faz duas coisas: x = x + atenção(normaliza(x)) e x = x + MLP(normaliza(x)). Em seguida vem a normalização final (ln_f), ainda com formato (B, T, C). Por fim, a camada de saída (lm_head) produz as notas, com formato (B, T, V).

Um ponto importante: o formato (B, T, C) é o mesmo do começo ao fim dos blocos. Por isso dá para empilhar 12 blocos, ou 48 no GPT-2 gigante, sem mudar nada.


3. COMPONENTES

3.1 Tokenização

O que faz: transforma texto em números, usando o BPE do GPT-2 (biblioteca tiktoken).

Como: o BPE começa com os 256 bytes e vai juntando os pares mais frequentes, até chegar a 50.257 tokens. Palavras comuns viram 1 token, e palavras raras viram vários.

Por que existe: o computador só trabalha com números. Usar letras deixaria as sequências longas demais, e usar palavras inteiras exigiria um vocabulário enorme.

Entrada e saída: texto entra, lista de inteiros sai. Por exemplo, "Hello, I'm a language model," vira 8 tokens.

3.2 Embeddings de token (wte) e de posição (wpe)

O que faz: troca cada número por uma lista de 768 números aprendidos, a ficha do token, e soma a ficha da posição.

Como: são tabelas de consulta (nn.Embedding). O wte tem 50257 linhas e o wpe tem 1024.

Por que a posição: a atenção sozinha não sabe a ordem das palavras. "O cachorro mordeu o homem" e "o homem mordeu o cachorro" teriam as mesmas fichas. Somar a ficha da posição resolve.

Entrada e saída: (B, T) inteiros viram (B, T, 768).

3.3 Auto-atenção causal (CausalSelfAttention)

É a parte em que os tokens trocam informação.

Como funciona: cada token gera três vetores com uma multiplicação de matriz (c_attn). A query diz o que o token está procurando. A key diz o que ele tem. O value é o que ele entrega se prestarem atenção nele.

A conta: multiplica a query de um token pela key de cada token anterior. Quanto maior o resultado, mais os dois combinam. O softmax transforma esses resultados em porcentagens que somam 100%, e o token recebe uma mistura dos values nessas proporções.

Exemplo: na frase "o gato dormiu", a palavra "dormiu" ficou com 23% de atenção em "o", 63% em "gato" e 14% nela mesma. Ela puxa mais informação do sujeito.

Máscara causal: cada token só vê os anteriores. Sem isso, no treino o modelo veria a resposta (o próximo token) e não aprenderia nada. Com a máscara, uma sequência de T tokens vira T exemplos de treino ao mesmo tempo.

Divisão por raiz de 64: sem ela, os números ficam grandes, o softmax dá quase 100% para um token só e o modelo deixa de aprender.

Várias cabeças (12): são 12 atenções em paralelo, de 64 números cada, que podem aprender relações diferentes. No código, isso é só reorganizar o tensor com view e transpose.

FlashAttention: o código usa F.scaled_dot_product_attention, que faz a mesma conta, mas sem guardar a matriz T por T inteira na memória. É mais rápido.

Entrada e saída: (B, T, 768) entra e (B, T, 768) sai.

3.4 MLP

O que faz: processa cada token sozinho: 768 vira 3072, passa pelo filtro GELU e volta para 768.

Por que existe: a atenção reúne informação, e o MLP trabalha em cima dela. Sem a GELU, que não é linear, duas camadas seguidas seriam equivalentes a uma só.

GELU em vez de ReLU: ela é suave e não zera completamente os valores negativos. Uso a versão com tanh, que é a do GPT-2 original.

Entrada e saída: (B, T, 768) entra e (B, T, 768) sai.

3.5 Bloco (Block) e conexão residual

No código, o bloco faz x = x + self.attn(self.ln_1(x)) e depois x = x + self.mlp(self.ln_2(x)).

Residual: cada parte só soma uma correção ao que já existe, em vez de substituir. Na volta do treino, o sinal de erro consegue passar pela soma direto até as primeiras camadas.

LayerNorm antes de cada parte: mantém os números de cada token numa faixa estável, com média 0 e espalhamento 1.

Relação com o Transformer original: o GPT-2 usa só o decoder, e a normalização vem antes das partes, não depois.

3.6 Modelo completo (GPT)

Saída (lm_head): transforma a ficha final de cada posição em 50.257 notas, uma por token possível.

Compartilhamento de pesos: o wte e o lm_head usam a mesma matriz. Isso economiza cerca de 38,6 milhões de parâmetros, uns 30% do modelo, e faz sentido porque tokens parecidos devem ter fichas parecidas e notas parecidas.

Inicialização: pesos com média 0 e desvio 0,02. As camadas que escrevem no fluxo residual (c_proj) recebem um fator extra de 1 sobre raiz de (2 vezes o número de camadas). Cada bloco soma duas contribuições ao fluxo, então sem esse fator os números cresceriam com a profundidade.

Loss: cross-entropy entre as notas e o token correto, que é a entrada deslocada em 1. Com pesos aleatórios, o esperado é o logaritmo natural de 50257, cerca de 10,82.

Contagem de parâmetros: meu código dá 124.439.808, o mesmo número do GPT-2 oficial.

3.7 Treino (train_gpt2.py)

O AdamW ajusta o passo de cada peso automaticamente.

O weight decay, aplicado só nas matrizes, puxa os pesos levemente para zero, evitando depender de poucos pesos enormes.

O warmup com cosseno usa passos pequenos no começo e depois suavemente menores até o fim.

O gradient clipping (limite 1.0) corta correções gigantes que estragariam o modelo.

O gradient accumulation simula um lote grande somando vários lotes pequenos. A loss é dividida pelo número de lotes, porque a cross-entropy já tira a média.

O bf16 e o TF32 usam números menos precisos para ganhar velocidade e memória, mas só com placa de vídeo.

Cada passo de treino faz isto: zerar os gradientes, calcular a loss, chamar backward() (que calcula como cada peso influencia o erro), aplicar o clipping e chamar optimizer.step() (que ajusta os pesos).

3.8 Geração (GPT.generate)

Primeiro, roda o modelo no texto atual. Segundo, pega só as notas da última posição. Terceiro, divide pela temperatura: baixa é previsível, alta é criativa. Quarto, aplica o top-k, que só deixa concorrer os k tokens mais prováveis. Quinto, sorteia um token, acrescenta ao texto e repete.


4. RELAÇÃO COM OS CONTEÚDOS DE AULA

[AJUSTE esta parte aos termos e exemplos que o professor usou em aula.]

O modelo de linguagem e a probabilidade condicional aparecem no forward, em que cada posição calcula a probabilidade do próximo token dado os anteriores, e no generate, que sorteia dessa distribuição. A representação vetorial está no wte. O mecanismo de atenção com Q, K e V está na CausalSelfAttention. A rede feed-forward e a função de ativação estão no MLP e na GELU. A normalização é o LayerNorm. As conexões residuais, que combatem o gradiente que some, são o x = x + algo. Softmax e cross-entropy estão em F.cross_entropy e F.softmax. O aprendizado auto-supervisionado está em x = buf[:-1] e y = buf[1:]. Backpropagation e gradiente descendente são o loss.backward() e o optimizer.step(). A regularização é o weight decay. A transferência de aprendizado é o from_pretrained.


5. TESTES E RESULTADOS

5.1 O código está correto?

O python model.py deu 124.439.808 parâmetros, logits com formato (2, 16, 50257) e loss inicial de 11,17. O generate.py com os pesos oficiais da OpenAI gerou texto coerente. O generate.py com o meu checkpoint gerou falas com formato de peça de teatro.

A loss inicial de 11,17 ficou um pouco acima de 10,82. Isso é normal: 10,82 só vale para previsões exatamente uniformes, e com poucos tokens aleatórios a loss varia alguns décimos. Valores como 15 ou 20 indicariam inicialização errada.

O teste mais forte foi carregar os pesos da OpenAI no meu código e gerar texto coerente. Se a arquitetura estivesse errada, os nomes e formatos dos pesos não encaixariam, ou o texto sairia sem sentido.

5.2 Treino

Com o modelo pequeno (4 camadas, 256 dimensões) e 200 passos, a loss caiu de cerca de 11 para cerca de 6. O texto gerado tem palavras reais e formato de peça, mas as frases não têm sentido, o que é esperado para tão pouco treino.

5.3 Experimentos: tirando peças

Usei o mesmo modelo, 150 passos e a mesma semente. Mudei uma peça por vez. Os números abaixo são loss de treino, loss de validação e o maior valor do gradiente.

Modelo normal: 5,43, 5,64 e 6,4.
Sem posição: 5,25, 5,52 e 7,3.
Sem máscara: 5,36, 5,64 e 14,1.
Sem residual: 6,39, 6,54 e 17,0.

Sem residual foi o único efeito claro: a loss ficou cerca de 1 ponto pior e o gradiente ficou instável. Isso confirma o papel do atalho x = x + algo no treino.

Sem máscara, esperava que a loss de treino despencasse, com o modelo colando o futuro, mas ela ficou igual à normal. O gradiente máximo dobrou. Minha hipótese é que 150 passos são poucos para o modelo aprender a copiar. Não provei isso.

Sem posição, a loss ficou até um pouco menor. Minhas hipóteses são que, com poucos passos, a posição ajuda pouco, e que a máscara causal já dá uma noção parcial de ordem, porque cada token vê um número diferente de anteriores. Também pode ser ruído, pois rodei cada variante uma vez só.

Limitações: uma execução por variante e poucos passos. Só a diferença do "sem residual" é grande o suficiente para eu confiar nela. Para os outros dois, seria preciso rodar mais passos e repetir.


6. REFLEXÃO INDIVIDUAL

[ESTA SEÇÃO É SUA. Escreva com suas palavras e com o que você realmente viveu. Não deixei texto pronto de propósito.]

Perguntas para guiar. Qual parte foi mais difícil de entender, e o que fez você entender (um exemplo, um print, o vídeo)? Que dificuldades apareceram? Que partes te surpreenderam? O que você estudaria a seguir? Como você usou a IA e o que aprendeu com isso?

Fatos reais da execução, para usar se quiser: o python model.py deu erro no Windows por causa do caractere "≈" (UnicodeEncodeError) e foi corrigido trocando por "~"; o treino em CPU ficou lento, cerca de 4 segundos por passo; e dois dos experimentos não deram o resultado que a teoria previa.


7. CONCLUSÃO

[escreva 1 ou 2 parágrafos seus]


REFERÊNCIAS

Karpathy, A. Let's reproduce GPT-2 (124M). 2024.
Vaswani et al. Attention Is All You Need. 2017.
Radford et al. Language Models are Unsupervised Multitask Learners. OpenAI, 2019.
Press e Wolf. Using the Output Embedding to Improve Language Models. 2017.
Dao et al. FlashAttention. 2022.
