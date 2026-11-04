# Resultados dos experimentos (rodados de verdade, na sua máquina)

Modelo pequeno (4 camadas, 256 números por token), 150 passos de treino, mesma semente em todos. Mudei **uma peça por vez**. Menor loss = melhor.

| variante | loss treino | loss validação | maior norm do gradiente |
|---|---|---|---|
| normal | 5,43 | 5,64 | 6,4 |
| sem posição | 5,25 | 5,52 | 7,3 |
| sem máscara | 5,36 | 5,64 | 14,1 |
| sem residual | **6,39** | **6,54** | 17,0 |

- **Loss de treino:** o erro nos textos que o modelo estudou.
- **Loss de validação:** o erro em um trecho que ele nunca viu, com a máscara ligada, que é como o modelo é usado de verdade ao gerar texto.
- **Norm:** o tamanho do "grito de correção" (gradiente). Muito alto significa treino instável.

## O que cada resultado quer dizer

### Sem residual: o efeito claro
A loss ficou **~1 ponto pior** que a do modelo normal, e o gradiente chegou a 17, o maior de todos. Sem o `x = x + ...`, o sinal de erro precisa atravessar todas as camadas sem atalho. Ele fica bagunçado, e o modelo aprende pior e de forma instável. Esse é o único experimento em que a peça removida fez uma diferença grande, e confirma a explicação da seção 5 do `EXPLICACAO_SIMPLES.md`.

### Sem máscara: o gradiente ficou instável, mas a "trapaça" não apareceu
Esperava que a loss de treino despencasse (o modelo "colando" o futuro). **Não aconteceu**: ficou 5,36, quase igual à do normal. O gradiente máximo dobrou (14,1 contra 6,4), o que mostra que algo ficou mais instável.

Meu palpite é que 150 passos são poucos para o modelo descobrir como copiar a próxima palavra. Isso é uma hipótese, não algo que eu provei. Para testar, rode `python experimentos.py 1500` e veja se a loss de treino do "sem_mascara" cai muito abaixo da validação. Se cair, a trapaça apareceu.

### Sem posição: ficou até um pouco melhor
Contra a expectativa, a loss foi **menor** (5,25 contra 5,43). Duas explicações possíveis, que também são palpites:
- Com poucos passos, o modelo ainda está aprendendo as palavras mais comuns, e a posição quase não ajuda.
- A atenção com máscara já carrega uma noção parcial de ordem: a 1ª palavra só vê uma, a 10ª vê dez.

Também pode ser ruído, porque cada variante foi rodada uma única vez. Uma diferença de 0,2 com uma execução só não prova nada.

## Limites desses testes
- Cada variante rodou **uma vez** só, e 150 passos é pouco. Diferenças pequenas (como a de "sem posição") podem ser sorte.
- Só o "sem residual" tem uma diferença grande o bastante para eu confiar nela.
- Em um relatório, dizer "esperava X, obtive Y, e minhas hipóteses são Z" mostra mais entendimento do que fingir que tudo deu certo.

## Sugestão para você
Rode de novo com mais passos (`python experimentos.py 600`, uns 10 minutos por variante) e compare. Se o resultado mudar, isso vira um ótimo trecho da sua reflexão.
