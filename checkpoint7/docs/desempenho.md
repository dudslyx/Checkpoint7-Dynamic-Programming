# Desempenho — medições reais

Tempo: `time.perf_counter()` (melhor de N repetições, sem tracemalloc). Memória: pico do `tracemalloc` em uma execução separada. Gerado por `scripts/gerar_docs.py` a partir de `questao_0X/resultados/bench_*.csv`.
Máquina de geração: Python 3.12.3 em Linux (x86_64).

## Questão 1

### Brute Force — pequena / média / grande (m = SKUs do subconjunto)
| m_skus | combinacoes | tempo_s | pico_kib | combinacoes_por_s |
|---|---|---|---|---|
| 5 | 16,807 | 0.0005 | 526 | 34,380,619 |
| 6 | 84,035 | 0.0030 | 2,627 | 27,969,675 |
| 7 | 420,175 | 0.0172 | 13,131 | 24,448,489 |
| 8 | 2,100,875 | 0.1588 | 32,769 | 13,229,102 |
| 9 | 10,504,375 | 0.7557 | 32,769 | 13,900,176 |

Razão de tempo entre tamanhos consecutivos: [6.1, 5.7, 9.2, 4.8] → crescimento exponencial (cada SKU a mais multiplica a busca por ≈ K). Para os 27 SKUs ativos do catálogo completo seriam **2.32e+14 combinações ≈ 1.6 anos** nesta máquina: inviável → GA.

### Algoritmo Genético — escalabilidade (G=100 gerações fixas, sem parada antecipada)
| variavel | n | P | G | tempo_s | pico_kib | avaliacoes_por_s |
|---|---|---|---|---|---|---|
| n (SKUs) | 27 | 120 | 100 | 0.0187 | 245 | 641,852 |
| n (SKUs) | 108 | 120 | 100 | 0.0517 | 909 | 231,939 |
| n (SKUs) | 432 | 120 | 100 | 0.1829 | 3,409 | 65,605 |
| n (SKUs) | 1728 | 120 | 100 | 0.7486 | 13,410 | 16,029 |
| P (população) | 27 | 40 | 100 | 0.0099 | 87 | 405,192 |
| P (população) | 27 | 120 | 100 | 0.0188 | 245 | 639,957 |
| P (população) | 27 | 360 | 100 | 0.0440 | 707 | 818,067 |

Inclinação log-log do tempo: **0.89** em n (esperado ≈ 1, O(G·P·n)) e **0.68** em P (esperado ≈ 1).

### Greedy (200 lotes)
| n_skus | lotes | tempo_s | pico_kib |
|---|---|---|---|
| 30 | 200 | 0.0032 | 20 |
| 300 | 200 | 0.0122 | 59 |
| 3,000 | 200 | 0.0975 | 461 |

Inclinação log-log em n: **0.74** (esperado ≈ 1, O((n+L) log n)).

### Otimização de código — antes × depois
| alvo | versao_antes | versao_depois | tempo_antes_s | tempo_depois_s | speedup | pico_antes_kib | pico_depois_kib |
|---|---|---|---|---|---|---|---|
| Brute Force (m=6, 84035 combinações) | itertools.product + recálculo em Python | tabelas pré-computadas + decodificação vetorizada em blocos | 0.5101 | 0.0027 | 185.9x | 0 | 2,627 |
| Fitness do GA (300 avaliações de população 120x27) | laços Python (soma gene a gene) | gather vetorizado numpy | 0.0447 | 0.0051 | 8.8x | 1,157 | 370 |

Troca explícita: a versão otimizada gasta mais memória (tabelas e blocos) para ganhar tempo.

## Questão 2

### Três tamanhos de entrada (n SKUs; instância replicada com dívidas, caixa e orçamento escalados)
| n_skus | etapa | tempo_s | pico_kib | cenarios |
|---|---|---|---|---|
| 24 | Monte Carlo — estimativa | 0.0282 | 10,615 | 2,000 |
| 24 | Programação Dinâmica | 0.0386 | 3,275 | — |
| 24 | Greedy semanal | 0.0291 | 40 | — |
| 24 | Monte Carlo — simulação | 0.0216 | 1,780 | 1,000 |
| 96 | Monte Carlo — estimativa | 0.1198 | 42,260 | 2,000 |
| 96 | Programação Dinâmica | 0.1644 | 7,101 | — |
| 96 | Greedy semanal | 0.1025 | 149 | — |
| 96 | Monte Carlo — simulação | 0.0873 | 6,561 | 1,000 |
| 384 | Monte Carlo — estimativa | 0.5099 | 168,892 | 2,000 |
| 384 | Programação Dinâmica | 0.6200 | 22,404 | — |
| 384 | Greedy semanal | 0.4186 | 532 | — |
| 384 | Monte Carlo — simulação | 0.3160 | 25,685 | 1,000 |

Inclinações log-log do tempo em n: Greedy semanal: 0.96, Monte Carlo — estimativa: 1.04, Monte Carlo — simulação: 0.97, Programação Dinâmica: 1.00.

### Número de cenários (simulação de validação, n = 24 SKUs)
| n_cenarios | tempo_s | pico_kib | cenarios_por_s |
|---|---|---|---|
| 1,000 | 0.0219 | 1,780 | 45,633 |
| 2,000 | 0.0396 | 1,897 | 50,457 |
| 5,000 | 0.0994 | 2,249 | 50,306 |
| 1e+04 | 0.2004 | 2,836 | 49,902 |

Inclinação log-log em N: **0.97** (esperado ≈ 1, O(N·W·n)). Memória quase constante em N porque os cenários são simulados em blocos de 500.

### Reutilização de subproblemas na DP
42,120 subproblemas distintos (semana × estoque, somados sobre os SKUs) foram resolvidos uma única vez e lidos 10,108,800 vezes (240 leituras por estado = níveis × quantis); a enumeração de trajetórias testaria L^W = 2,176,782,336 caminhos de preço por SKU.
