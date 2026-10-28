# Avaliação Varejo — Checkpoint 6 (Dynamic Programming) — Grupo G01

> ⚠️ **INSTÂNCIA PROVISÓRIA — NÃO ENTREGAR ASSIM.** Os arquivos em `data/` foram reconstruídos a partir dos prints do notebook (grupo assumido: G01). Gere os arquivos oficiais com o notebook do professor (Bloco 02 preenchido), copie-os para `data/`, apague os provisórios e rode `python run_all.py`. Este README e todos os números abaixo são regerados automaticamente e o aviso some.

**Grupo: G01**

| Nome | RM |
|---|---|
| Eduardo Francisco Mauro Gonçalves | 561969 |
| Igor Dominiski | 562055 |
| Murillo Akira | 561886 |

- **Seed base:** 20
- **Seed final:** 1020
- **ID da instância:** 6b77ba9e0f93
- Hashes SHA-256 dos arquivos: `data/manifesto_dados.json` (verificados automaticamente antes de qualquer execução).

Problema: recuperação financeira de varejistas por precificação online. Q1 — Operação Caixa 90 (Brute Force + Algoritmo Genético + Greedy); Q2 — Plano de Sobrevivência em 12 Semanas (Monte Carlo + Programação Dinâmica + Greedy).

## Como reproduzir

```bash
pip install -r requirements.txt
# 1) gerar os 5 arquivos com o notebook oficial e copiá-los para data/
python run_all.py                                   # verifica SHA-256, roda Q1 e Q2, regera README e docs
python questao_01/main.py --seed 1020 --modo completo
python questao_02/main.py --seed 1020 --cenarios 5000
python -m unittest discover -s tests -t .           # 3 módulos de teste (também roda com pytest)
```

Ambiente em que os resultados foram gerados: Python 3.12.3, numpy 2.4.4, pandas 3.0.2. Todas as fontes de aleatoriedade usam fluxos independentes derivados da seed (`comum/aleatorio.py`); mesma seed → mesmos resultados.

**Cenário-surpresa (por parâmetro, sem editar código):** `--meta-fator 1.10`, `--concorrente-delta -0.08`, `--custo-logistico-fator 1.15`, `--orcamento-fator 0.75`, `--divida-fator 1.2`, `--sku-indisponivel Q1-007`, `--ga-populacao 200`, `--ga-mutacao 0.08`. Ver `docs/guia_defesa.md`.

## Nesta questão utilizamos 3 algoritmos — Questão 1

1. **Brute Force** enumera todas as combinações de preços permitidos de um subconjunto de 8 SKUs (2,100,875 combinações) e entrega o **ótimo exato**.
2. **Algoritmo Genético** roda no mesmo subconjunto (validação: gap vs Brute Force) e depois no catálogo completo, produzindo a **política de preços**.
3. **Greedy** recebe a política do GA e distribui o orçamento de incentivo pelo maior benefício marginal por real investido.

Fluxo: `Brute Force ⇄ GA (validação)` → `GA (catálogo completo)` → `Greedy` → resultado final. Diagrama: `docs/arquitetura.png`.

## Nesta questão utilizamos 3 algoritmos — Questão 2

1. **Monte Carlo** estima a distribuição da demanda por SKU e nível de preço (≥ 2.000 cenários; fontes: demanda, preço do concorrente, custo logístico) → quantis que alimentam a DP.
2. **Programação Dinâmica** (estado = semana × estoque restante; decisão = nível de preço) devolve a política de reprecificação semanal e a trajetória ótima.
3. **Greedy** aloca por semana o recurso promocional escasso, guiado pelo plano e pelo valor futuro do estoque da DP. O **plano completo volta ao Monte Carlo** (5.000 cenários independentes) para validar o risco.

Fluxo: `Monte Carlo → DP → Greedy → Monte Carlo (validação)`.

## Resultados — Questão 1 (`questao_01/resultados/`)

A meta de caixa de 90 dias **foi atingida**: caixa final R$ 345.740,38 contra meta R$ 219.809,58 → folga de R$ 125.930,80. Sem o Greedy a meta já seria atingida.

**Tabela Brute Force × GA no subconjunto** (`tabela_bf_vs_ga.csv`; gap médio do GA-A: 0.000000%):

| metodo | caixa_rs | gap_pct | tempo_s | pico_kib | observacao |
|---|---|---|---|---|---|
| Brute Force (m=8 SKUs) | R$ 162.909,15 | 0.0000% | 0.2050 | 32,769 | 2100875 combinações avaliadas (ótimo exato) |
| GA A (principal) (seed principal) | R$ 162.909,15 | 0.0000% | 0.0061 | 79 | 48 gerações, 5760 avaliações |
| GA A (principal) — 10 execuções | — | 0.0000% | 0.0060 | — | gap médio; pior gap 0.0000%; achou o ótimo em 100% das execuções |
| GA B (enxuta) (seed principal) | R$ 162.909,15 | 0.0000% | 0.0066 | 30 | 69 gerações, 2760 avaliações |
| GA B (enxuta) — 10 execuções | — | 0.0000% | 0.0052 | — | gap médio; pior gap 0.0000%; achou o ótimo em 100% das execuções |

**Duas configurações do GA no catálogo completo** (`comparacao_configuracoes_ga.csv`; gráfico de fitness por geração: `convergencia_ga.png`):

| configuracao | populacao | mutacao_por_gene | elitismo | caixa_rs | gap_vs_otimo_exato_pct | gap_medio_pct_nas_repeticoes | geracoes | geracao_99pct | tempo_s | pico_kib |
|---|---|---|---|---|---|---|---|---|---|---|
| A (principal) | 120 | 0.05 | 4 | R$ 341.660,05 | 0.0000% | 0.0000% | 60 | 12 | 0.0114 | 244 |
| B (enxuta) | 40 | 0.01 | 1 | R$ 341.660,05 | 0.0000% | 0.0306% | 92 | 28 | 0.0092 | 86 |

**Antes/depois** (`antes_depois_greedy.csv`):

| etapa | caixa_rs | margem_pct | estoque_remanescente_un | incentivo_gasto_rs | cobertura_divida_pct | folga_meta_rs | meta_atingida |
|---|---|---|---|---|---|---|---|
| 0. Preços atuais (nível 1,00) | R$ 217.957,18 | 10,69% | 698 | R$ 0,00 | 112,60% | R$ -1.852,40 | False |
| 1. Após GA (política de preços) | R$ 341.660,05 | 19,26% | 1,127 | R$ 0,00 | 176,51% | R$ 121.850,47 | True |
| 2. Após Greedy (incentivos) | R$ 345.740,38 | 18,83% | 1,034 | R$ 12.424,81 | 178,62% | R$ 125.930,80 | True |

Interpretação financeira:

- **Ganho da política de preços:** o caixa sai de R$ 217.957,18 (preços atuais) para R$ 341.660,05 com o GA (56,8%), porque o modelo eleva preço onde a margem é fina e a demanda é inelástica ou limitada por estoque, respeitando o piso de margem de 5,40%.
- **SKUs suspensos:** 3 SKU(s) não têm nenhum preço permitido (Q1-022, Q1-023, Q1-024); ficam fora da venda em vez de serem liquidados abaixo do piso.
- **Cobertura da dívida:** 178,6% da dívida de R$ 193.563,10 (saldo após a dívida: R$ 152.177,28).
- **Incentivos:** o Greedy usou 13,0% do orçamento (R$ 12.424,81 de R$ 95.575,47) em 9 SKU(s); parou porque o benefício marginal por real caiu abaixo de R$ 1,00 — gastar mais destruiria caixa.
- **Estoque remanescente:** 1,034 un. (R$ 780.582,95 ao custo; R$ 350.963,56 com mais de 180 dias — risco de obsolescência).

## Resultados — Questão 2 (`questao_02/resultados/`)

**Probabilidade de pagamento no prazo** (`prob_pagamento.csv`, 5000 cenários de validação independentes):

| compromisso | valor_rs | P_pagar_no_prazo | IC95_inf | IC95_sup | P_falha | limite_falha | dentro_do_limite |
|---|---|---|---|---|---|---|---|
| semana 4 | R$ 56.155,94 | 100,00% | 100,00% | 100,00% | 0,00% | 10% | True |
| semana 8 | R$ 72.534,75 | 100,00% | 100,00% | 100,00% | 0,00% | 10% | True |
| semana 12 | R$ 105.292,38 | 100,00% | 100,00% | 100,00% | 0,00% | 10% | True |
| as três juntas | R$ 233.983,07 | 100,00% | — | — | 0,00% | 10% | True |

**Comparação de políticas** (mesmos cenários — números aleatórios comuns — `comparativo_politicas.csv`):

| politica | P_pagar_todas | caixa_acumulado_medio_rs | saldo_liquido_medio_rs | saldo_liquido_P10_rs | margem_pct | estoque_final_pct | cenarios_falha |
|---|---|---|---|---|---|---|---|
| Base: preço atual fixo (nível 1,00) | 100,00% | R$ 331.710,06 | R$ 97.726,99 | R$ 94.537,46 | 15,44% | 11,64% | 0 |
| DP — trajetória fixa (malha aberta), sem promoção | 100,00% | R$ 501.079,52 | R$ 267.096,45 | R$ 260.082,77 | 22,81% | 18,54% | 0 |
| DP — reprecificação semanal adaptativa, sem promoção | 100,00% | R$ 502.776,21 | R$ 268.793,14 | R$ 261.802,33 | 22,88% | 18,64% | 0 |
| PLANO COMPLETO: DP adaptativa + Greedy | 100,00% | R$ 508.108,34 | R$ 274.125,27 | R$ 269.053,41 | 22,72% | 15,87% | 0 |
| Plano completo sob choque PERSISTENTE do concorrente | 100,00% | R$ 486.030,05 | R$ 252.046,98 | R$ 222.817,95 | 22,48% | 19,46% | 0 |

**Teste de estresse — plano NÃO replanejado** (`estresse.csv`):

| cenario | P_todas_base | P_todas_plano | saldo_medio_base_rs | saldo_medio_plano_rs | plano_dentro_do_limite |
|---|---|---|---|---|---|
| Cenário atual | 100,00% | 100,00% | R$ 97.726,99 | R$ 274.125,27 | True |
| Concorrente −8% | 100,00% | 100,00% | R$ 79.648,44 | R$ 245.530,64 | True |
| Concorrente −15% | 100,00% | 100,00% | R$ 61.597,73 | R$ 213.898,84 | True |
| Custo logístico +15% | 100,00% | 100,00% | R$ 81.632,26 | R$ 258.627,18 | True |
| Custo logístico +30% | 100,00% | 100,00% | R$ 65.537,53 | R$ 243.129,09 | True |
| Dívidas +20% | 100,00% | 100,00% | R$ 50.930,37 | R$ 227.328,66 | True |
| Dívidas +40% | 93,18% | 100,00% | R$ 4.133,76 | R$ 180.532,05 | True |
| Dívidas +60% | 0,00% | 100,00% | R$ -42.662,85 | R$ 133.735,43 | True |
| Dívidas +80% | 0,00% | 100,00% | R$ -89.459,47 | R$ 86.938,82 | True |
| Dívidas +100% | 0,00% | 100,00% | R$ -136.256,08 | R$ 40.142,20 | True |
| Combinado: concorrente −8%, logística +15%, dívidas +20% | 100,00% | 100,00% | R$ 17.421,82 | R$ 184.083,02 | True |

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
