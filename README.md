# Checkpoint7-Dynamic-Programming
Calibração da urgência θ da DP (cenários de construção; `calibracao_theta.csv`): | theta | P_pagar_todas | caixa_acumulado_medio_rs | saldo_liquido_P10_rs | pesos_semana | escolhido |
|---|---|---|---|---|---|
| 0 | 1 | 5.082e+05 | 2.694e+05 | 1.00,1.00,1.00 | True |
| 0.5 | 1 | 5.061e+05 | 2.676e+05 | 1.50,1.25,1.00 | False |
| 1 | 1 | 4.994e+05 | 2.612e+05 | 2.00,1.50,1.00 | False |
| 2 | 1 | 4.909e+05 | 2.529e+05 | 3.00,2.00,1.00 | False |

Evidências gráficas: `distribuicao_caixa_final.png`, `trajetoria_preco_estoque.png` (+ `.csv`), `plano_semanal.csv` (todos os SKUs × semanas).

Interpretação financeira:

- **Risco de pagamento:** com o plano completo, P(pagar no prazo) = semana 4: 100,0%, semana 8: 100,0%, semana 12: 100,0%; as três juntas: 100,0% (0 falhas em 5000 cenários). Limite de falha da instância: 10% → dentro do limite.
- **Caixa:** saldo líquido médio (caixa acumulado − dívidas) de R$ 274.125,27; no percentil 10 (pior 10% dos cenários) R$ 269.053,41; CVaR R$ 266.765,42. Margem operacional 22,7%; estoque final 15,9% do inicial.
- **Valor da política:** o preço fixo atual gera caixa acumulado médio de R$ 331.710,06; o plano gera R$ 508.108,34 (53,2%).
- **Folga de segurança:** como a probabilidade já é alta no cenário atual, o teste de estresse mostra a robustez: o plano suporta dívidas até ×2.15 mantendo P(pagar tudo) ≥ 90%, contra ×1.40 do preço fixo.
- **Promoções (Greedy):** gasto de R$ 12.095,27 de R$ 53.756,76 disponíveis (22,5%).

## Onde cada resultado é produzido

| Resultado | Arquivo gerado | Função |
|---|---|---|
| Ótimo exato (subconjunto) | `questao_01/resultados/tabela_bf_vs_ga.csv` | `questao_01/brute_force.py::brute_force` |
| Gap do GA | `tabela_bf_vs_ga.csv`, `ga_vs_bf_repeticoes.csv` | `genetic_algorithm.py::executar_ga` |
| Fitness por geração | `convergencia_ga.png/.csv` | `executar_ga` (históricos) + `comum/graficos.py::convergencia` |
| Política de preços | `politica_precos.csv` | `questao_01/main.py::executar` → `modelo.py::avaliar_politica` |
| Antes/depois Greedy | `antes_depois_greedy*.csv` | `questao_01/greedy.py::alocar_incentivos` |
| Meta, folga, cobertura da dívida | `resultado_final.json` | `questao_01/main.py::executar` |
| Estatísticas Monte Carlo | (internas, alimentam a DP) | `questao_02/monte_carlo.py::estimar_estatisticas` |
| Política/trajetória da DP | `plano_semanal.csv`, `trajetoria_preco_estoque.*` | `dynamic_programming.py::resolver_dp`, `reconstruir_caminho` |
| Promoções semanais | `plano_semanal.csv` | `questao_02/greedy.py::alocar_promocoes` |
| P(pagamento), distribuição de caixa | `prob_pagamento.csv`, `distribuicao_caixa_final.*` | `monte_carlo.py::simular`, `pagamentos`, `resumir` |
| Estresse / ruptura | `estresse.csv`, `resultado_final.json` | `questao_02/main.py::executar`, `ruptura` |
| Tempo e memória | `bench_*.csv`, `docs/desempenho.md` | `comum/metricas.py::medir`, `questao_0X/benchmark.py` |
| Otimização antes/depois | `questao_01/resultados/otimizacao_antes_depois.csv` | `questao_01/benchmark.py::otimizacao_antes_depois` |

## Documentação

`docs/complexidade.md` (Big O de tempo e memória), `docs/desempenho.md` (medições reais), `docs/decisoes.md` (modelagem e premissas), `docs/arquitetura.png`, `docs/guia_defesa.md`, `IA_USO.md`.

