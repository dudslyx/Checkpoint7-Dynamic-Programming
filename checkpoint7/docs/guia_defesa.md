# Guia de defesa e checklist de entrega

## Antes de entregar
1. Rode o notebook oficial com o **Bloco 02**: `GRUPO = "GXX"` (o seu código real) e `MATRICULAS = ["561969", "562055", "561886"]`. Use **Run all** (executar só o Bloco 11 dá `NameError: PASTA_SAIDA`, pois o Bloco 01 não rodou).
2. Copie para `data/` os 5 arquivos (`q1_catalogo_GXX.csv`, `q1_config_GXX.json`, `q2_catalogo_GXX.csv`, `q2_config_GXX.json`, `manifesto_dados.json`) e **apague os provisórios (`*_G01.*`)** se o seu grupo não for G01.
3. Atualize `equipe.json` se necessário e rode `python run_all.py` (README, resultados e docs são regerados; o aviso de “provisório” some).
4. Confira o README: grupo, seed base, seed final e ID da instância vêm do manifesto oficial.
5. Preencha `IA_USO.md` com o que o grupo realmente revisou/alterou.
6. Faça commits em etapas conforme for validando (sugestão de ordem): dados e manifesto → modelo e Brute Force → GA → Greedy Q1 → Monte Carlo → DP → Greedy Q2 → integração/`main.py` → testes → otimização/benchmarks → análise/docs.

## Perguntas típicas → onde está a resposta
- *Por que três algoritmos?* `docs/arquitetura.png` e README (dados trocados entre as etapas).
- *Reutilização de subproblemas na DP?* `dynamic_programming.py::resolver_dp`: `V[prox[l]]` é lido por todos os (estado, decisão, quantil); contadores em `resultado_final.json → dp`.
- *Por que o GA acerta o ótimo?* Domínio já filtrado pelo piso de margem; objetivo aditivo; veja `tabela_bf_vs_ga.csv`.
- *Por que o Greedy parou?* Benefício marginal por real < 1 (`comum/incentivos.py::alocar_guloso`).
- *Validação independente?* `monte_carlo.py::simular` com fluxos `calibracao` e `validacao`.

## Cenário-surpresa (tudo por parâmetro)
| Pedido do professor | Comando |
|---|---|
| meta de caixa +10% (Q1) | `python questao_01/main.py --meta-fator 1.10` |
| dívidas +10% (Q2) | `python questao_02/main.py --divida-fator 1.10` |
| concorrente −8% | `python questao_02/main.py --concorrente-delta -0.08` |
| custo logístico +15% | `--custo-logistico-fator 1.15` (Q1 e Q2) |
| orçamento promocional −25% | `--orcamento-fator 0.75` (Q1 e Q2) |
| SKU indisponível | `--sku-indisponivel Q1-007` (repetível) |
| população/mutação do GA | `python questao_01/main.py --ga-populacao 200 --ga-mutacao 0.08` |

Adicione `--modo rapido` para pular benchmarks durante a defesa.
