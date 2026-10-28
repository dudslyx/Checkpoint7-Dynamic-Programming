"""Gera README.md e docs/desempenho.md a partir dos resultados REAIS (nada digitado à mão)."""
import json
import platform
import sys
from pathlib import Path

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parents[1]
R1, R2 = RAIZ / "questao_01" / "resultados", RAIZ / "questao_02" / "resultados"


def brl(x, d=2):
    return "R$ " + f"{x:,.{d}f}".replace(",", "X").replace(".", ",").replace("X", ".")


def pct(x, d=1):
    return f"{x:.{d}f}%".replace(".", ",")


def md(df, fmt=None):
    fmt = fmt or {}
    cols = list(df.columns)
    out = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, r in df.iterrows():
        cel = []
        for c in cols:
            v = r[c]
            if c in fmt and pd.notna(v):
                cel.append(fmt[c](v))
            elif isinstance(v, (float, np.floating)):
                cel.append("—" if pd.isna(v) else f"{v:,.4g}")
            else:
                cel.append(str(v))
        out.append("| " + " | ".join(cel) + " |")
    return "\n".join(out)


def slope(x, y):
    x, y = np.log(np.asarray(x, float)), np.log(np.asarray(y, float))
    return float(np.polyfit(x, y, 1)[0])


def main():
    q1 = json.loads((R1 / "resultado_final.json").read_text(encoding="utf-8"))
    q2 = json.loads((R2 / "resultado_final.json").read_text(encoding="utf-8"))
    eq = json.loads((RAIZ / "equipe.json").read_text(encoding="utf-8"))["integrantes"]
    i1, m, d, g1, gc = q1["instancia"], q1["meta"], q1["divida"], q1["greedy"], q1["ga_completo"]
    i2, pc, pb = q2["instancia"], q2["plano_completo"], q2["base"]
    tbf = pd.read_csv(R1 / "tabela_bf_vs_ga.csv")
    cmp_ga = pd.read_csv(R1 / "comparacao_configuracoes_ga.csv")
    ad = pd.read_csv(R1 / "antes_depois_greedy.csv")
    cpol = pd.read_csv(R2 / "comparativo_politicas.csv")
    prob = pd.read_csv(R2 / "prob_pagamento.csv")
    est = pd.read_csv(R2 / "estresse.csv")
    cal = pd.read_csv(R2 / "calibracao_theta.csv")
    bfe, gae = pd.read_csv(R1 / "bench_brute_force.csv"), pd.read_csv(R1 / "bench_ga.csv")
    gre, otm = pd.read_csv(R1 / "bench_greedy.csv"), pd.read_csv(R1 / "otimizacao_antes_depois.csv")
    bi, bc = pd.read_csv(R2 / "bench_instancia.csv"), pd.read_csv(R2 / "bench_cenarios.csv")
    prov = i1["provisoria"] or i2["provisoria"]
    seed_cmd = i1["seed_execucao"]

    banner = ""
    if prov:
        banner = ("> ⚠️ **INSTÂNCIA PROVISÓRIA — NÃO ENTREGAR ASSIM.** Os arquivos em `data/` foram reconstruídos a partir dos prints do notebook "
                  "(grupo assumido: G01). Gere os arquivos oficiais com o notebook do professor (Bloco 02 preenchido), copie-os para `data/`, apague os "
                  "provisórios e rode `python run_all.py`. Este README e todos os números abaixo são regerados automaticamente e o aviso some.\n\n")

    sub = q1["brute_force"]
    ga_ok = q1["ga_validacao"]["gap_medio_A_pct"]
    texto_meta = (f"A meta de caixa de 90 dias **{'foi atingida' if m['atingida'] else 'NÃO foi atingida'}**: caixa final {brl(m['caixa_final'])} contra meta "
                  f"{brl(m['meta'])} → {'folga' if m['atingida'] else 'insuficiência'} de {brl(abs(m['folga_rs']))}.")
    texto_meta += (" Sem o Greedy a meta " + ("já seria atingida." if m["atingida_sem_greedy"] else "não seria atingida."))
    base_cx = q1["politica_base"]["caixa"]
    analise_q1 = (
        f"- **Ganho da política de preços:** o caixa sai de {brl(base_cx)} (preços atuais) para {brl(q1['politica_ga']['caixa'])} com o GA "
        f"({pct(100*(q1['politica_ga']['caixa']/base_cx-1))}), porque o modelo eleva preço onde a margem é fina e a demanda é inelástica ou limitada por estoque, "
        f"respeitando o piso de margem de {pct(i1['margem_minima_pct'],2)}.\n"
        f"- **SKUs suspensos:** {len(i1['skus_suspensos'])} SKU(s) não têm nenhum preço permitido ({', '.join(i1['skus_suspensos']) or 'nenhum'}); ficam fora da venda em vez de serem liquidados abaixo do piso.\n"
        f"- **Cobertura da dívida:** {pct(d['cobertura_pct'])} da dívida de {brl(d['total'])} (saldo após a dívida: {brl(d['saldo_apos_divida_rs'])}).\n"
        f"- **Incentivos:** o Greedy usou {pct(g1['uso_orcamento_pct'])} do orçamento ({brl(g1['gasto_total'])} de {brl(i1['orcamento_promocional'])}) em {g1['skus_com_incentivo']} SKU(s); "
        f"parou porque o benefício marginal por real caiu abaixo de R$ 1,00 — gastar mais destruiria caixa.\n"
        f"- **Estoque remanescente:** {q1['politica_final']['estoque_remanescente_un']:,.0f} un. ({brl(q1['politica_final']['estoque_remanescente_custo'])} ao custo; "
        f"{brl(q1['politica_final']['estoque_antigo_180d_custo'])} com mais de 180 dias — risco de obsolescência).")
    rb = q2["ruptura_multiplicador_dividas"]
    analise_q2 = (
        f"- **Risco de pagamento:** com o plano completo, P(pagar no prazo) = " + ", ".join(f"semana {s}: {pct(100*v)}" for s, v in pc["p_pagamento"].items()) +
        f"; as três juntas: {pct(100*pc['p_todas'])} ({pc['cenarios_falha']} falhas em {pc['n_cenarios']} cenários). Limite de falha da instância: {pct(100*i2['limite_risco_falha'],0)} → "
        f"{'dentro do limite' if pc['risco_ok'] else 'FORA do limite'}.\n"
        f"- **Caixa:** saldo líquido médio (caixa acumulado − dívidas) de {brl(pc['saldo_liquido_medio'])}; no percentil {int(pc['percentil_risco'])} (pior 10% dos cenários) {brl(pc['saldo_liquido_percentil'])}; "
        f"CVaR {brl(pc['cvar'])}. Margem operacional {pct(pc['margem_pct'])}; estoque final {pct(pc['estoque_final_pct'])} do inicial.\n"
        f"- **Valor da política:** o preço fixo atual gera caixa acumulado médio de {brl(pb['caixa_acumulado_medio'])}; o plano gera {brl(pc['caixa_acumulado_medio'])} "
        f"({pct(100*(pc['caixa_acumulado_medio']/pb['caixa_acumulado_medio']-1))}).\n"
        f"- **Folga de segurança:** como a probabilidade já é alta no cenário atual, o teste de estresse mostra a robustez: o plano suporta dívidas até ×{rb['plano']:.2f} mantendo P(pagar tudo) ≥ "
        f"{pct(100*(1-i2['limite_risco_falha']),0)}, contra ×{rb['base']:.2f} do preço fixo.\n"
        f"- **Promoções (Greedy):** gasto de {brl(q2['greedy']['gasto_total'])} de {brl(q2['greedy']['orcamento_total'])} disponíveis ({pct(q2['greedy']['uso_orcamento_pct'])}).")

    fmt_brl = lambda v: brl(v)
    fmt_pct = lambda v: pct(v, 2)
    readme = f"""# Avaliação Varejo — Checkpoint 6 (Dynamic Programming) — Grupo {i1['grupo']}

{banner}**Grupo: {i1['grupo']}**

| Nome | RM |
|---|---|
""" + "\n".join(f"| {p['nome']} | {p['rm']} |" for p in eq) + f"""

- **Seed base:** {i1['seed_base']}
- **Seed final:** {i1['seed_final']}
- **ID da instância:** {i1['id_instancia']}
- Hashes SHA-256 dos arquivos: `data/manifesto_dados.json` (verificados automaticamente antes de qualquer execução).

Problema: recuperação financeira de varejistas por precificação online. Q1 — Operação Caixa 90 (Brute Force + Algoritmo Genético + Greedy); Q2 — Plano de Sobrevivência em 12 Semanas (Monte Carlo + Programação Dinâmica + Greedy).

## Como reproduzir

```bash
pip install -r requirements.txt
# 1) gerar os 5 arquivos com o notebook oficial e copiá-los para data/
python run_all.py                                   # verifica SHA-256, roda Q1 e Q2, regera README e docs
python questao_01/main.py --seed {seed_cmd} --modo completo
python questao_02/main.py --seed {seed_cmd} --cenarios 5000
python -m unittest discover -s tests -t .           # {len(list((RAIZ/'tests').glob('test_*.py')))} módulos de teste (também roda com pytest)
```

Ambiente em que os resultados foram gerados: Python {platform.python_version()}, numpy {np.__version__}, pandas {pd.__version__}. Todas as fontes de aleatoriedade usam fluxos independentes derivados da seed (`comum/aleatorio.py`); mesma seed → mesmos resultados.

**Cenário-surpresa (por parâmetro, sem editar código):** `--meta-fator 1.10`, `--concorrente-delta -0.08`, `--custo-logistico-fator 1.15`, `--orcamento-fator 0.75`, `--divida-fator 1.2`, `--sku-indisponivel Q1-007`, `--ga-populacao 200`, `--ga-mutacao 0.08`. Ver `docs/guia_defesa.md`.

## Nesta questão utilizamos 3 algoritmos — Questão 1

1. **Brute Force** enumera todas as combinações de preços permitidos de um subconjunto de {sub['m_skus']} SKUs ({sub['combinacoes']:,} combinações) e entrega o **ótimo exato**.
2. **Algoritmo Genético** roda no mesmo subconjunto (validação: gap vs Brute Force) e depois no catálogo completo, produzindo a **política de preços**.
3. **Greedy** recebe a política do GA e distribui o orçamento de incentivo pelo maior benefício marginal por real investido.

Fluxo: `Brute Force ⇄ GA (validação)` → `GA (catálogo completo)` → `Greedy` → resultado final. Diagrama: `docs/arquitetura.png`.

## Nesta questão utilizamos 3 algoritmos — Questão 2

1. **Monte Carlo** estima a distribuição da demanda por SKU e nível de preço (≥ 2.000 cenários; fontes: demanda, preço do concorrente, custo logístico) → quantis que alimentam a DP.
2. **Programação Dinâmica** (estado = semana × estoque restante; decisão = nível de preço) devolve a política de reprecificação semanal e a trajetória ótima.
3. **Greedy** aloca por semana o recurso promocional escasso, guiado pelo plano e pelo valor futuro do estoque da DP. O **plano completo volta ao Monte Carlo** (5.000 cenários independentes) para validar o risco.

Fluxo: `Monte Carlo → DP → Greedy → Monte Carlo (validação)`.

## Resultados — Questão 1 (`questao_01/resultados/`)

{texto_meta}

**Tabela Brute Force × GA no subconjunto** (`tabela_bf_vs_ga.csv`; gap médio do GA-A: {ga_ok:.6f}%):

{md(tbf, {'caixa_rs': fmt_brl, 'gap_pct': lambda v: f'{v:.4f}%', 'tempo_s': lambda v: f'{v:.4f}', 'pico_kib': lambda v: f'{v:,.0f}'})}

**Duas configurações do GA no catálogo completo** (`comparacao_configuracoes_ga.csv`; gráfico de fitness por geração: `convergencia_ga.png`):

{md(cmp_ga[['configuracao','populacao','mutacao_por_gene','elitismo','caixa_rs','gap_vs_otimo_exato_pct','gap_medio_pct_nas_repeticoes','geracoes','geracao_99pct','tempo_s','pico_kib']], {'caixa_rs': fmt_brl, 'gap_vs_otimo_exato_pct': lambda v: f'{v:.4f}%', 'gap_medio_pct_nas_repeticoes': lambda v: f'{v:.4f}%', 'tempo_s': lambda v: f'{v:.4f}', 'pico_kib': lambda v: f'{v:,.0f}'})}

**Antes/depois** (`antes_depois_greedy.csv`):

{md(ad[['etapa','caixa_rs','margem_pct','estoque_remanescente_un','incentivo_gasto_rs','cobertura_divida_pct','folga_meta_rs','meta_atingida']], {'caixa_rs': fmt_brl, 'margem_pct': fmt_pct, 'incentivo_gasto_rs': fmt_brl, 'cobertura_divida_pct': fmt_pct, 'folga_meta_rs': fmt_brl, 'estoque_remanescente_un': lambda v: f'{v:,.0f}'})}

Interpretação financeira:

{analise_q1}

## Resultados — Questão 2 (`questao_02/resultados/`)

**Probabilidade de pagamento no prazo** (`prob_pagamento.csv`, {pc['n_cenarios']} cenários de validação independentes):

{md(prob, {'valor_rs': fmt_brl, 'P_pagar_no_prazo': lambda v: pct(100*v, 2), 'IC95_inf': lambda v: pct(100*v, 2), 'IC95_sup': lambda v: pct(100*v, 2), 'P_falha': lambda v: pct(100*v, 2), 'limite_falha': lambda v: pct(100*v, 0)})}

**Comparação de políticas** (mesmos cenários — números aleatórios comuns — `comparativo_politicas.csv`):

{md(cpol[['politica','P_pagar_todas','caixa_acumulado_medio_rs','saldo_liquido_medio_rs','saldo_liquido_P10_rs','margem_pct','estoque_final_pct','cenarios_falha']], {'P_pagar_todas': lambda v: pct(100*v,2), 'caixa_acumulado_medio_rs': fmt_brl, 'saldo_liquido_medio_rs': fmt_brl, 'saldo_liquido_P10_rs': fmt_brl, 'margem_pct': fmt_pct, 'estoque_final_pct': fmt_pct})}

**Teste de estresse — plano NÃO replanejado** (`estresse.csv`):

{md(est, {'P_todas_base': lambda v: pct(100*v,2), 'P_todas_plano': lambda v: pct(100*v,2), 'saldo_medio_base_rs': fmt_brl, 'saldo_medio_plano_rs': fmt_brl})}

Calibração da urgência θ da DP (cenários de construção; `calibracao_theta.csv`): {md(cal)}

Evidências gráficas: `distribuicao_caixa_final.png`, `trajetoria_preco_estoque.png` (+ `.csv`), `plano_semanal.csv` (todos os SKUs × semanas).

Interpretação financeira:

{analise_q2}

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
"""
    (RAIZ / "README.md").write_text(readme, encoding="utf-8")

    # ---- desempenho.md -------------------------------------------------------------------------------------------
    gn = gae[gae["variavel"] == "n (SKUs)"]
    gp = gae[gae["variavel"] == "P (população)"]
    s_ga_n, s_ga_p = slope(gn["n"], gn["tempo_s"]), slope(gp["P"], gp["tempo_s"])
    s_bf = slope(bfe["m_skus"][:0] + 1, bfe["tempo_s"][:0] + 1) if False else None
    cresc = (bfe["tempo_s"].iloc[1:].to_numpy() / bfe["tempo_s"].iloc[:-1].to_numpy()).round(1).tolist()
    s_gr = slope(gre["n_skus"], gre["tempo_s"])
    q2i = bi.pivot(index="n_skus", columns="etapa", values="tempo_s")
    slopes_q2 = {c: slope(q2i.index, q2i[c]) for c in q2i.columns}
    s_mc = slope(bc["n_cenarios"], bc["tempo_s"])
    est_bf = q1.get("benchmark", {}).get("estimativa_brute_force_catalogo", {})
    des = f"""# Desempenho — medições reais

Tempo: `time.perf_counter()` (melhor de N repetições, sem tracemalloc). Memória: pico do `tracemalloc` em uma execução separada. Gerado por `scripts/gerar_docs.py` a partir de `questao_0X/resultados/bench_*.csv`.
Máquina de geração: Python {platform.python_version()} em {platform.system()} ({platform.machine()}).

## Questão 1

### Brute Force — pequena / média / grande (m = SKUs do subconjunto)
{md(bfe, {'combinacoes': lambda v: f'{v:,.0f}', 'tempo_s': lambda v: f'{v:.4f}', 'pico_kib': lambda v: f'{v:,.0f}', 'combinacoes_por_s': lambda v: f'{v:,.0f}'})}

Razão de tempo entre tamanhos consecutivos: {cresc} → crescimento exponencial (cada SKU a mais multiplica a busca por ≈ K). Para os {i1['skus_ativos']} SKUs ativos do catálogo completo seriam **{est_bf.get('combinacoes_catalogo_completo', float('nan')):.2e} combinações ≈ {est_bf.get('anos_estimados', float('nan')):.2g} anos** nesta máquina: inviável → GA.

### Algoritmo Genético — escalabilidade (G=100 gerações fixas, sem parada antecipada)
{md(gae, {'tempo_s': lambda v: f'{v:.4f}', 'pico_kib': lambda v: f'{v:,.0f}', 'avaliacoes_por_s': lambda v: f'{v:,.0f}'})}

Inclinação log-log do tempo: **{s_ga_n:.2f}** em n (esperado ≈ 1, O(G·P·n)) e **{s_ga_p:.2f}** em P (esperado ≈ 1).

### Greedy (200 lotes)
{md(gre, {'tempo_s': lambda v: f'{v:.4f}', 'pico_kib': lambda v: f'{v:,.0f}'})}

Inclinação log-log em n: **{s_gr:.2f}** (esperado ≈ 1, O((n+L) log n)).

### Otimização de código — antes × depois
{md(otm, {'tempo_antes_s': lambda v: f'{v:.4f}', 'tempo_depois_s': lambda v: f'{v:.4f}', 'speedup': lambda v: f'{v:.1f}x', 'pico_antes_kib': lambda v: f'{v:,.0f}', 'pico_depois_kib': lambda v: f'{v:,.0f}'})}

Troca explícita: a versão otimizada gasta mais memória (tabelas e blocos) para ganhar tempo.

## Questão 2

### Três tamanhos de entrada (n SKUs; instância replicada com dívidas, caixa e orçamento escalados)
{md(bi, {'tempo_s': lambda v: f'{v:.4f}', 'pico_kib': lambda v: f'{v:,.0f}'})}

Inclinações log-log do tempo em n: """ + ", ".join(f"{k}: {v:.2f}" for k, v in slopes_q2.items()) + f""".

### Número de cenários (simulação de validação, n = {i2['n_skus']} SKUs)
{md(bc, {'tempo_s': lambda v: f'{v:.4f}', 'pico_kib': lambda v: f'{v:,.0f}', 'cenarios_por_s': lambda v: f'{v:,.0f}'})}

Inclinação log-log em N: **{s_mc:.2f}** (esperado ≈ 1, O(N·W·n)). Memória quase constante em N porque os cenários são simulados em blocos de 500.

### Reutilização de subproblemas na DP
{q2['dp']['estados']:,} subproblemas distintos (semana × estoque, somados sobre os SKUs) foram resolvidos uma única vez e lidos {q2['dp']['leituras_V']:,} vezes ({q2['dp']['reuso_por_estado']:.0f} leituras por estado = níveis × quantis); a enumeração de trajetórias testaria L^W = {q2['dp']['caminhos_ingenuos_L_pow_W']:,.0f} caminhos de preço por SKU.
"""
    (RAIZ / "docs" / "desempenho.md").write_text(des, encoding="utf-8")
    print("README.md e docs/desempenho.md gerados.")


if __name__ == "__main__":
    main()
