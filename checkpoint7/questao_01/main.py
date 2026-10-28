"""Questão 1 — Operação Caixa 90 (Brute Force -> validação do GA -> GA completo -> Greedy).

Exemplos:
    python questao_01/main.py --seed 1020 --modo completo
    python questao_01/main.py --modo rapido --meta-fator 1.10 --ga-populacao 200
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from comum import graficos
from comum.aleatorio import criar_rng
from comum.cenarios import adicionar_argumentos, ajustes_de_args
from comum.dados import carregar_dados
from comum.metricas import medir
from questao_01 import benchmark
from questao_01.brute_force import brute_force
from questao_01.genetic_algorithm import CONFIG_A, CONFIG_B, com_parametros, executar_ga
from questao_01.greedy import alocar_incentivos
from questao_01.modelo import (avaliar_politica, construir_dominio, construir_instancia, construir_tabelas,
                               escolher_subconjunto, genes_para_niveis, nivel_referencia, totais)

PASTA = Path(__file__).resolve().parent


def _py(o):
    if isinstance(o, dict): return {str(k): _py(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)): return [_py(v) for v in o]
    if isinstance(o, np.ndarray): return _py(o.tolist())
    if isinstance(o, (np.integer,)): return int(o)
    if isinstance(o, (np.floating,)): return float(o)
    if isinstance(o, (np.bool_,)): return bool(o)
    return o


def _gap(otimo, valor):
    return 100.0 * (otimo - valor) / abs(otimo) if otimo else 0.0


def executar(dados, ajustes, seed, modo="completo", bf_skus=8, repeticoes=10, ga_pop=None, ga_mut=None, saida=None) -> dict:
    saida = Path(saida) if saida else PASTA / "resultados"
    saida.mkdir(parents=True, exist_ok=True)
    inst = construir_instancia(dados.q1_catalogo, dados.q1_config, ajustes)
    tab = construir_tabelas(inst)
    dom = construir_dominio(tab)
    n = inst.n
    cfg_a, cfg_b = com_parametros(CONFIG_A, ga_pop, ga_mut), CONFIG_B

    # ---- Etapa A: Brute Force no subconjunto reduzido ------------------------------------------------
    linhas = escolher_subconjunto(dom, bf_skus, criar_rng(seed, "q1-subconjunto"))
    sub = dom.subconjunto(linhas)
    mbf = medir(brute_force, sub.caixa, sub.k)
    bf = mbf.resultado

    # ---- Etapa B1: GA no MESMO subconjunto (validação contra o ótimo exato) ---------------------------
    rep = []
    tab_bf = [{"metodo": f"Brute Force (m={len(sub.k)} SKUs)", "caixa_rs": bf.melhor_caixa, "gap_pct": 0.0, "tempo_s": mbf.tempo_s,
               "pico_kib": mbf.pico_kib, "observacao": f"{bf.combinacoes} combinações avaliadas (ótimo exato)"}]
    for cfg in (cfg_a, cfg_b):
        gaps, tempos, gens = [], [], []
        for r in range(repeticoes):
            fluxo = f"q1-ga-sub-{cfg.nome[0]}"
            if r == 0:
                med = medir(lambda: executar_ga(sub.caixa, sub.k, cfg, criar_rng(seed, fluxo)))
                res, t, pico = med.resultado, med.tempo_s, med.pico_kib
                tab_bf.append({"metodo": f"GA {cfg.nome} (seed principal)", "caixa_rs": res.melhor_caixa,
                               "gap_pct": _gap(bf.melhor_caixa, res.melhor_caixa), "tempo_s": t, "pico_kib": pico,
                               "observacao": f"{res.geracoes_executadas} gerações, {res.avaliacoes} avaliações"})
            else:
                t0 = time.perf_counter()
                res = executar_ga(sub.caixa, sub.k, cfg, criar_rng(seed + r, fluxo))
                t = time.perf_counter() - t0
            gaps.append(_gap(bf.melhor_caixa, res.melhor_caixa)); tempos.append(t); gens.append(res.geracoes_executadas)
            rep.append({"config": cfg.nome, "execucao": r, "seed": seed + r if r else seed, "caixa_rs": res.melhor_caixa,
                        "gap_pct": gaps[-1], "geracoes": res.geracoes_executadas, "tempo_s": t})
        tab_bf.append({"metodo": f"GA {cfg.nome} — {repeticoes} execuções", "caixa_rs": np.nan, "gap_pct": float(np.mean(gaps)),
                       "tempo_s": float(np.mean(tempos)), "pico_kib": np.nan,
                       "observacao": f"gap médio; pior gap {max(gaps):.4f}%; achou o ótimo em {100*np.mean(np.array(gaps) < 1e-9):.0f}% das execuções"})
    pd.DataFrame(tab_bf).to_csv(saida / "tabela_bf_vs_ga.csv", index=False)
    pd.DataFrame(rep).to_csv(saida / "ga_vs_bf_repeticoes.csv", index=False)

    # ---- Etapa B2: GA no catálogo completo (2 configurações) -----------------------------------------
    otimo_exato = float(dom.caixa.max(axis=1).sum()) if len(dom.k) else 0.0   # objetivo aditivo => ótimo por decomposição
    comp, hist, execs = [], {}, {}
    for cfg in (cfg_a, cfg_b):
        fluxo = f"q1-ga-{cfg.nome[0]}"
        med = medir(lambda: executar_ga(dom.caixa, dom.k, cfg, criar_rng(seed, fluxo)))
        r0 = med.resultado
        gaps = [_gap(otimo_exato, executar_ga(dom.caixa, dom.k, cfg, criar_rng(seed + 1 + r, fluxo)).melhor_caixa) for r in range(repeticoes)]
        alvo = 0.99 * r0.melhor_caixa
        g99 = int(np.argmax(r0.hist_melhor >= alvo)) + 1
        comp.append({"configuracao": cfg.nome, "populacao": cfg.populacao, "mutacao_por_gene": cfg.p_mutacao, "torneio_k": cfg.torneio_k,
                     "cruzamento": cfg.p_cruzamento, "elitismo": cfg.elitismo, "paciencia": cfg.paciencia,
                     "caixa_rs": r0.melhor_caixa, "gap_vs_otimo_exato_pct": _gap(otimo_exato, r0.melhor_caixa),
                     "gap_medio_pct_nas_repeticoes": float(np.mean(gaps)), "geracoes": r0.geracoes_executadas,
                     "geracao_99pct": g99, "avaliacoes": r0.avaliacoes, "tempo_s": med.tempo_s, "pico_kib": med.pico_kib})
        hist[cfg.nome], execs[cfg.nome] = (r0.hist_melhor, r0.hist_media), r0
    pd.DataFrame(comp).to_csv(saida / "comparacao_configuracoes_ga.csv", index=False)
    gmax = max(len(v[0]) for v in hist.values())
    conv = pd.DataFrame({"geracao": np.arange(1, gmax + 1)})
    for nome, (mel, med_) in hist.items():
        conv[f"{nome[0]}_melhor"] = np.pad(mel, (0, gmax - len(mel)), constant_values=np.nan)
        conv[f"{nome[0]}_media"] = np.pad(med_, (0, gmax - len(med_)), constant_values=np.nan)
    conv.to_csv(saida / "convergencia_ga.csv", index=False)
    graficos.convergencia(saida / "convergencia_ga.png", hist, "Fitness (caixa 90 dias) por geração — catálogo completo")

    melhor_cfg = max(execs, key=lambda k: (execs[k].melhor_caixa, k == cfg_a.nome))
    ga = execs[melhor_cfg]
    niveis = genes_para_niveis(dom, ga.melhores_genes, n)

    # ---- Etapa C: Greedy sobre a política do GA -------------------------------------------------------
    df_ga = avaliar_politica(inst, tab, niveis)
    gr = alocar_incentivos(inst, tab, df_ga)
    df_base = avaliar_politica(inst, tab, np.full(n, nivel_referencia(inst)))
    t_base, t_ga, t_fim = totais(df_base), totais(df_ga), totais(df_ga.assign(
        caixa=gr.detalhe["caixa_depois"] + gr.gasto_por_sku, unidades=gr.detalhe["unidades_depois"],
        receita=gr.detalhe["unidades_depois"] * df_ga["preco"], estoque_restante=gr.detalhe["estoque_rest_depois"]), gr.gasto_total)

    def linha(nome, t):
        return {"etapa": nome, "caixa_rs": t["caixa"], "receita_bruta_rs": t["receita_bruta"], "margem_pct": 100 * t["margem_pct"],
                "unidades_vendidas": t["unidades_vendidas"], "estoque_remanescente_un": t["estoque_remanescente_un"],
                "estoque_remanescente_custo_rs": t["estoque_remanescente_custo"], "incentivo_gasto_rs": t["gasto_incentivo"],
                "cobertura_divida_pct": 100 * t["caixa"] / inst.divida, "folga_meta_rs": t["caixa"] - inst.meta,
                "meta_atingida": t["caixa"] >= inst.meta}
    antes_depois = pd.DataFrame([linha("0. Preços atuais (nível 1,00)", t_base), linha("1. Após GA (política de preços)", t_ga),
                                 linha("2. Após Greedy (incentivos)", t_fim)])
    antes_depois.to_csv(saida / "antes_depois_greedy.csv", index=False)
    gr.detalhe.to_csv(saida / "antes_depois_greedy_por_sku.csv", index=False)
    pol = df_ga.drop(columns=["caixa", "receita", "unidades", "estoque_restante"]).copy()
    pol.insert(4, "preco_atual", inst.preco)
    pol = pol.join(gr.detalhe[["unidades_antes", "caixa_antes", "incentivo_rs", "unidades_depois", "caixa_depois", "estoque_rest_depois"]])
    pol.to_csv(saida / "politica_precos.csv", index=False)

    final = {
        "instancia": {"grupo": dados.grupo, "seed_base": dados.seed_base, "seed_final": dados.seed_final, "seed_execucao": seed,
                      "id_instancia": dados.id_instancia, "provisoria": dados.provisorio, "n_skus": n, "skus_ativos": int(len(dom.ativos)),
                      "skus_suspensos": inst.sku[~np.isin(np.arange(n), dom.ativos)].tolist(), "ajustes": ajustes.descricao(),
                      "meta_caixa_90d": inst.meta, "divida_total_90d": inst.divida, "orcamento_promocional": inst.orcamento,
                      "margem_minima_pct": 100 * inst.margem_min, "niveis": inst.niveis.tolist()},
        "brute_force": {"m_skus": int(len(sub.k)), "skus": inst.sku[sub.ativos].tolist(), "combinacoes": int(bf.combinacoes),
                        "caixa_otimo": bf.melhor_caixa, "tempo_s": mbf.tempo_s, "pico_kib": mbf.pico_kib},
        "ga_validacao": {"gap_medio_A_pct": float(tab_bf[2]["gap_pct"]), "gap_medio_B_pct": float(tab_bf[4]["gap_pct"]),
                         "gap_seed_principal_A_pct": float(tab_bf[1]["gap_pct"]), "gap_seed_principal_B_pct": float(tab_bf[3]["gap_pct"])},
        "ga_completo": {"configuracao_usada": melhor_cfg, "caixa": ga.melhor_caixa, "otimo_exato_decomposicao": otimo_exato,
                        "gap_vs_otimo_exato_pct": _gap(otimo_exato, ga.melhor_caixa), "geracoes": ga.geracoes_executadas},
        "politica_base": t_base, "politica_ga": t_ga, "politica_final": t_fim,
        "greedy": {"gasto_total": gr.gasto_total, "lotes": gr.lotes, "uso_orcamento_pct": 100 * gr.gasto_total / inst.orcamento if inst.orcamento else 0.0,
                   "skus_com_incentivo": int((gr.gasto_por_sku > 0).sum())},
        "meta": {"meta": inst.meta, "caixa_final": t_fim["caixa"], "folga_rs": t_fim["caixa"] - inst.meta, "atingida": bool(t_fim["caixa"] >= inst.meta),
                 "caixa_sem_greedy": t_ga["caixa"], "atingida_sem_greedy": bool(t_ga["caixa"] >= inst.meta)},
        "divida": {"total": inst.divida, "cobertura_pct": 100 * t_fim["caixa"] / inst.divida, "saldo_apos_divida_rs": t_fim["caixa"] - inst.divida},
    }

    # ---- Benchmarks (modo completo) -------------------------------------------------------------------
    if modo == "completo":
        bfe, est = benchmark.bf_escala(dom, seed, (5, 6, 7, 8, 9))
        gae = benchmark.ga_escala(dom, seed)
        gre = benchmark.greedy_escala(inst, seed=seed)
        tbe = benchmark.tabelas_escala(inst, seed=seed)
        otm = benchmark.otimizacao_antes_depois(inst, dom, seed)
        for nome, df in [("bench_brute_force", bfe), ("bench_ga", gae), ("bench_greedy", gre), ("bench_tabelas", tbe), ("otimizacao_antes_depois", otm)]:
            df.to_csv(saida / f"{nome}.csv", index=False)
        graficos.escala(saida / "bench_brute_force.png", bfe["m_skus"], {"tempo (s)": bfe["tempo_s"]}, "m (SKUs no subconjunto)", "tempo (s)",
                        "Brute Force: crescimento exponencial", logy=True)
        gn = gae[gae["variavel"] == "n (SKUs)"]
        graficos.escala(saida / "bench_ga.png", gn["n"], {"tempo (s)": gn["tempo_s"]}, "n (SKUs)", "tempo (s)",
                        "GA: tempo x n (P=120, G=100) — linear", logx=True, logy=True)
        final["benchmark"] = {"estimativa_brute_force_catalogo": est}
    (saida / "resultado_final.json").write_text(json.dumps(_py(final), ensure_ascii=False, indent=2), encoding="utf-8")
    return final


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    adicionar_argumentos(p)
    p.add_argument("--seed", type=int, default=None, help="seed dos componentes aleatórios (padrão: seed_final da instância)")
    p.add_argument("--modo", choices=["rapido", "completo"], default="completo")
    p.add_argument("--bf-skus", type=int, default=8, help="SKUs do subconjunto do Brute Force (>= 6)")
    p.add_argument("--repeticoes", type=int, default=10, help="execuções do GA por configuração (estatística)")
    p.add_argument("--ga-populacao", type=int, default=None)
    p.add_argument("--ga-mutacao", type=float, default=None)
    a = p.parse_args(argv)
    dados = carregar_dados(a.dados, verificar=not a.ignorar_integridade, permitir_provisorio=a.permitir_provisorio)
    seed = dados.seed_final if a.seed is None else a.seed
    r = executar(dados, ajustes_de_args(a), seed, a.modo, a.bf_skus, a.repeticoes, a.ga_populacao, a.ga_mutacao, a.saida)
    i, m, d = r["instancia"], r["meta"], r["divida"]
    print(f"[Q1] grupo {i['grupo']} | seed {seed} | instância {i['id_instancia']}" + (" | PROVISÓRIA" if i["provisoria"] else ""))
    print(f"[Q1] BF m={r['brute_force']['m_skus']}: ótimo R$ {r['brute_force']['caixa_otimo']:,.2f} | gap médio GA-A {r['ga_validacao']['gap_medio_A_pct']:.4f}%")
    print(f"[Q1] caixa final R$ {m['caixa_final']:,.2f} | meta R$ {m['meta']:,.2f} | {'FOLGA' if m['atingida'] else 'INSUFICIÊNCIA'} R$ {abs(m['folga_rs']):,.2f}")
    print(f"[Q1] cobertura da dívida {d['cobertura_pct']:.1f}% | orçamento usado {r['greedy']['uso_orcamento_pct']:.1f}%")
    return r


if __name__ == "__main__":
    main()
