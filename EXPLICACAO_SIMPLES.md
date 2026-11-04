GPT-2 EXPLICADO DO JEITO MAIS SIMPLES POSSÍVEL

Este texto é para entender. Não é o relatório. Leia, rode o código, brinque com ele e depois escreva o seu relatório com as suas palavras. Se algo aqui não ficar claro, o texto falhou, não você.


1. O QUE O GPT-2 FAZ

Uma coisa só: adivinha a próxima palavra (na verdade, o próximo pedacinho de palavra).

É igual ao corretor do celular quando você digita "bom" e ele sugere "dia". O GPT-2 faz isso, só que olhando para um texto inteiro (até 1024 pedacinhos) e não só para a última palavra.

Para escrever um texto, ele adivinha o próximo pedaço, cola no texto, adivinha o seguinte, cola, e assim vai.


2. O CAMINHO DE UMA FRASE DENTRO DO MODELO

Vamos seguir a frase "o gato dormiu". O caminho é: texto, números, vetores, 12 blocos, pontuações, próxima palavra.

Passo 1, virar números (tokenização). Computador só entende número. Então cada pedaço de texto vira um número, usando uma tabela com 50.257 itens. "o gato dormiu" vira algo como 78, 21443, 9203. Esses pedacinhos se chamam tokens. Palavras comuns são 1 token. Palavras raras são quebradas em vários.

Passo 2, virar vetores (embedding). Um número solto, como 21443, não diz nada. Então o modelo troca cada número por uma lista de 768 números, que é um vetor. Pense nisso como uma ficha de identidade do token, com 768 características. Essa ficha não é fixa: o modelo vai ajustando durante o treino. Palavras parecidas, como "gato" e "cachorro", acabam com fichas parecidas. No código isso é o wte, que é literalmente uma tabela: você dá o número da linha e ela devolve a lista.

Passo 3, dizer a posição. Se eu só entregar as fichas, o modelo não sabe a ordem. "O cachorro mordeu o homem" e "o homem mordeu o cachorro" teriam as mesmas fichas. A solução é ter outra tabela, com uma ficha para cada posição (1ª, 2ª, 3ª e assim por diante). Somamos a ficha da posição à ficha da palavra. É como colocar um número de fila em cada pessoa. No código é o wpe, e depois a linha x = tok_emb + pos_emb.

Passo 4, os 12 blocos. Aqui acontece o "pensar". A frase passa pelos 12 blocos em sequência. Cada bloco tem duas partes. A primeira é a atenção, em que as palavras conversam entre si. A segunda é o MLP, em que cada palavra pensa sozinha sobre o que ouviu. Os detalhes vêm abaixo.

Passo 5, virar pontuações. No final, a ficha da última palavra passa por uma camada que dá uma nota para cada um dos 50.257 tokens possíveis. A nota mais alta é o palpite do modelo para a próxima palavra. No código essa camada é o lm_head.


3. ATENÇÃO: A CONVERSA ENTRE PALAVRAS

Considere a frase "A Maria disse que ela estava cansada". Para entender o "ela", o modelo precisa olhar para "Maria". A atenção é o mecanismo que permite isso.

A analogia da sala de aula: cada palavra faz três coisas. Ela levanta uma pergunta (chamada Query), por exemplo "quem eu estou procurando?". Ela mostra uma etiqueta (chamada Key), que diz "eu sou isso aqui". E ela guarda um conteúdo (chamado Value), que é o que ela entrega se for escolhida.

A palavra "ela" faz a pergunta e compara com a etiqueta de todas as palavras anteriores. "Maria" tem a etiqueta que mais combina. Então "ela" pega bastante do conteúdo de "Maria" e pouco das outras.

Como a comparação funciona: multiplicar dois vetores, o que se chama produto escalar, dá um número. Quanto maior o número, mais parecidos eles são. É só isso. Depois o softmax transforma essas notas em porcentagens que somam 100%.

Um exemplo numérico: para a palavra "dormiu" na frase "o gato dormiu", os pesos podem ser 23% em "o", 63% em "gato" e 14% nela mesma. Ou seja, "dormiu" prestou mais atenção no "gato". A saída é uma mistura dos conteúdos nessas proporções.

A máscara, ou seja, proibido olhar para a frente. Quando o modelo treina, ele tenta adivinhar a próxima palavra. Se pudesse ver a próxima palavra, ele colaria e não aprenderia nada. Então cobrimos o futuro. A 1ª palavra só vê ela mesma. A 2ª vê a 1ª e ela mesma. E assim por diante. No código isso é o is_causal=True.

Por que dividir por raiz de 64? Os números do produto escalar crescem conforme o tamanho dos vetores. Se ficarem muito grandes, o softmax dá 100% para uma palavra só e 0% para as outras, e o modelo para de aprender. Dividir por raiz de 64, que é 8, esfria os números. É só um ajuste de volume.

Por que 12 cabeças? Em vez de uma conversa só, são 12 conversas em paralelo, cada uma com 64 números. Uma pode prestar atenção em quem é o sujeito, outra na palavra anterior, e assim por diante. No final juntamos tudo. No código isso é só reorganizar o tensor com view e transpose. Nada mágico.


4. MLP: CADA PALAVRA PENSA SOZINHA

Depois de ouvir as outras palavras, cada palavra passa por uma pequena rede neural. Ela estica os 768 números para 3072, passa por um filtro chamado GELU e volta para 768.

Esticar dá espaço de manobra para o modelo combinar características. O GELU é o filtro que dá ao modelo a capacidade de aprender coisas curvas e não só retas. Sem ele, empilhar camadas seria o mesmo que ter uma só. E aqui não há conversa entre palavras: cada uma trabalha por conta própria.

Resumindo: atenção é trocar informação, e MLP é processar a informação.


5. OS DOIS TRUQUES QUE MANTÊM TUDO FUNCIONANDO

Conexão residual, que no código é x = x + algo. Em vez de substituir a informação, o bloco só soma uma correção a ela. Pense que você está editando um texto: em vez de reescrever tudo a cada revisão, você só anota as mudanças em cima do original. Se uma revisão for ruim, o original continua lá. Isso também ajuda no treino, porque o sinal de erro consegue voltar da saída até o começo do modelo sem se perder pelo caminho. Sem isso, as camadas profundas quase não aprendem. Você viu esse efeito no experimento "sem residual".

LayerNorm. Ela mantém os números de cada palavra numa faixa razoável, com média 0 e espalhamento 1. Sem isso, os números podem crescer ou encolher a cada camada até virar bagunça. É como regular o volume para não ficar nem mudo nem estourando.


6. COMO O MODELO APRENDE (TREINO)

Primeiro, pega um trecho do texto, por exemplo 128 tokens. Segundo, o alvo é o mesmo trecho deslocado uma posição: para cada token, a resposta certa é o token seguinte. O texto é a própria prova com gabarito. Terceiro, o modelo dá seus palpites. Quarto, calcula-se o erro, chamado de loss. Se o modelo deu 90% para a palavra certa, o erro é pequeno. Se deu 0,1%, é grande. Quinto, vem a backpropagation, que é o loss.backward(). Ela calcula, para cada um dos 124 milhões de números do modelo, se mexer nele um pouquinho faz o erro subir ou descer. Sexto, o otimizador, que é o optimizer.step(), mexe cada número um pouquinho na direção que diminui o erro. Depois repete tudo milhares de vezes.

É como aprender a jogar dardos: joga, vê onde caiu, corrige a mira, joga de novo.

Sobre a loss: no começo o modelo chuta no escuro e a loss fica perto de 11, porque o logaritmo natural de 50257 é cerca de 10,8. Com treino ela cai. No seu teste foi para cerca de 6. Quanto menor, melhor. Um GPT-2 completo chega perto de 3.

Agora as técnicas do train_gpt2.py, uma por uma.

Learning rate é o tamanho de cada passo de correção. Se for grande demais, o modelo erra o alvo. Se for pequeno demais, demora muito.

Warmup significa começar com passos pequenos e ir aumentando, porque no início o modelo está perdido e erra feio.

Cosseno é diminuir os passos suavemente depois do warmup, para fazer ajustes finos no final.

Gradient clipping é cortar o ajuste quando ele é gigante. Isso evita que um exemplo estranho estrague o modelo.

Gradient accumulation serve para quando o computador não aguenta um lote grande de uma vez. Você divide em lotes pequenos e soma os resultados.

Weight decay é um puxão leve dos pesos para perto de zero, para o modelo não depender de poucos números gigantes.

bf16 é usar números menos precisos, mais arredondados, para ir mais rápido e gastar menos memória. Só faz diferença com placa de vídeo.


7. COMO O TEXTO É GERADO

Você dá um início, por exemplo "ROMEO:". O modelo dá uma nota para cada token possível. Depois entram dois ajustes.

A temperatura: se for baixa, o modelo escolhe quase sempre o mais provável, e o texto fica seguro e repetitivo. Se for alta, ele dá chance a escolhas esquisitas, e o texto fica criativo e bagunçado.

O top-k: só os k tokens mais prováveis podem concorrer. Isso evita sortear palavras absurdas.

Então o modelo sorteia um token, cola no texto e repete.


8. MINI-DICIONÁRIO

Token é um pedacinho de texto, uma palavra ou parte dela.
Embedding é a ficha de números de um token.
Parâmetro, ou peso, é cada número que o modelo ajusta no treino. São 124 milhões.
Logit é a nota crua para cada token possível.
Softmax é o que transforma notas em porcentagens que somam 100%.
Loss é o tamanho do erro do modelo.
Gradiente é a direção em que cada peso deve mudar para diminuir o erro.
Batch é quantos exemplos o modelo vê de uma vez.
Epoch é uma volta completa por todos os dados.


9. TESTE SE VOCÊ ENTENDEU (RESPONDA SEM OLHAR)

1. Por que o modelo precisa de embedding de posição?
2. O que a máscara causal impede? O que aconteceria sem ela no treino?
3. Qual a diferença entre o que a atenção faz e o que o MLP faz?
4. Por que o x = x + algo ajuda o treino?
5. Por que a loss inicial deve ser perto de 10,8?
6. Para que serve dividir a loss por grad_accum_steps?

Se travar em alguma, volte à seção correspondente e tente de novo. Entender de verdade é o que vai fazer diferença na entrega.


10. CONECTANDO COM OS EXPERIMENTOS

Os resultados reais da sua máquina estão no arquivo RESULTADOS.md. Cada experimento tira uma peça e mostra o que quebra. É a melhor prova de que você entendeu para que cada peça existe.
