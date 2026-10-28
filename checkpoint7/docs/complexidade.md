# Complexidade (Big O) — tempo e memória

Medições reais correspondentes em `docs/desempenho.md` (inclinações log-log medidas vs. esperadas).

## Variáveis
`n` = nº de SKUs ativos · `L` = nº de níveis de preço · `K` = preços permitidos por SKU (K ≤ L) · `m` = SKUs do subconjunto do Brute Force ·
`P` = população do GA · `G` = gerações · `t` = tamanho do torneio · `E` = lotes do Greedy · `W` = semanas (12) · `S` = estoque inicial máximo de um SKU ·
`Q` = quantis da demanda por SKU/nível (K_QUANTIS = 40) · `N` = cenários de Monte Carlo · `B` = bloco de cenários (500).

## Questão 1
| Algoritmo | Tempo | Memória | Trecho que domina | Quando deixa de ser viável → alternativa |
|---|---|---|---|---|
| Pré-cálculo das tabelas | O(n·L) | O(n·L) | `modelo.py::construir_tabelas` (arrays numpy) | nunca é gargalo; evita recalcular demanda no GA/BF |
| Brute Force | O(m·K^m) | O(m·K + bloco) | laço de blocos em `brute_force.py::brute_force` (decodificação de base mista + soma por SKU) | K^m explode: ~10⁷ combinações/s; com 27 SKUs ≈ 10¹⁴ → usa-se GA só como validação em m pequeno |
| Algoritmo Genético | O(G·P·n) (+ O(G·P·t) seleção) | O(P·n + n·K) | `genetic_algorithm.py::executar_ga` — gather da tabela, cruzamento e mutação, todos O(P·n) por geração | cresce linear; se n·P·G ficar grande, reduzir P ou usar parada por paciência |
| Greedy de incentivos | O((n + E)·log n) | O(n) | heap em `comum/incentivos.py::alocar_guloso` | ótimo para retornos côncavos separáveis (relaxação contínua); com restrições acopladas passaria a heurística |

## Questão 2
| Algoritmo | Tempo | Memória | Trecho que domina | Quando deixa de ser viável → alternativa |
|---|---|---|---|---|
| Monte Carlo (estimativa) | O(N·n·L) + ordenação O(n·L·N log N) dos quantis | O(N·n·L) | `monte_carlo.py::estimar_estatisticas` | N·n·L muito grande → amostrar em blocos / reduzir L |
| Programação Dinâmica | O(n·W·L·S·Q) | O(n·W·S) (política int8 + V) + O(S·Q·L) temporário por SKU | laço de semanas × níveis em `dynamic_programming.py::resolver_dp` (gather `V[prox]`) | estado por SKU é exato; o produto cartesiano de estoques de todos os SKUs seria O(Π S_i) → por isso a decomposição por SKU; se S crescer muito, agrupar estoque em faixas |
| Greedy semanal | O(W·(n + E)·Q·log n) | O(W·n) | `greedy.py::alocar_promocoes` (valor esperado por lote) | linear em n e W |
| Monte Carlo (validação) | O(N·W·n) | O(B·n + N·W) | laço semanal de `simular` | linear em N; blocos de 500 mantêm a memória estável |

**Reutilização de subproblemas (DP):** V(w+1, s') é calculado uma vez e lido por todos os (s, l, k) que levam a s'. Há W·(S+1) subproblemas por SKU contra L^W = 6^12 ≈ 2,2·10⁹ trajetórias de preço da enumeração (L=6).
