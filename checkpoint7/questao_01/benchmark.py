"""Medições reais (perf_counter + tracemalloc) para a Questão 1: escalabilidade e otimização antes/depois."""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from comum.aleatorio import criar_rng
from comum.metricas import medir
from questao_01.brute_force import brute_force, brute_force_ingenuo
from questao_01.genetic_algorithm import ConfigGA, avaliar_populacao, avaliar_populacao_ingenua, executar_ga
from questao_01.greedy import alocar_incentivos
from questao_01.modelo import (ampliar, avaliar_politica, caixa_sku_ingenuo, construir_dominio, construir_tabelas,
                               dados_ingenuos, escolher_subconjunto, nivel_referencia)


def bf_escala(dom, seed, tamanhos=(5, 6, 7, 8)):
    linhas = []
    for m in tamanhos:
        if m > len(dom.ativos):
            continue
        sub = dom.subconjunto(escolher_subconjunto(dom, m, criar_rng(seed, "bench-bf")))
        med = medir(brute_force, sub.caixa, sub.k, repeticoes=2 if m <= 7 else 1)
        comb = med.resultado.combinacoes
        linhas.append({"m_skus": m, "combinacoes": comb, "tempo_s": med.tempo_s, "pico_kib": med.pico_kib,
                       "combinacoes_por_s": comb / med.tempo_s})
    df = pd.DataFrame(linhas)
    est = {}
    if len(df):
        total_cheio = math.prod(int(v) for v in dom.k)
        taxa = float(df["combinacoes_por_s"].iloc[-1]) / max(1, int(df["m_skus"].iloc[-1])) * 1.0
        est = {"combinacoes_catalogo_completo": float(total_cheio),
               "anos_estimados": float(total_cheio * len(dom.k) / (df["combinacoes_por_s"].iloc[-1] * df["m_skus"].iloc[-1]) / 3.15e7)}
    return df, est


def ga_escala(dom, seed, fatores=(1, 4, 16, 64), pops=(40, 120, 360), geracoes=100):
    linhas = []
    base_cx, base_k = dom.caixa, dom.k
    for f in fatores:
        cx, k = np.tile(base_cx, (f, 1)), np.tile(base_k, f)
        cfg = ConfigGA(populacao=120, geracoes=geracoes, paciencia=None)
        med = medir(lambda: executar_ga(cx, k, cfg, criar_rng(seed, "bench-ga")), repeticoes=2)
        linhas.append({"variavel": "n (SKUs)", "n": len(k), "P": 120, "G": geracoes, "tempo_s": med.tempo_s, "pico_kib": med.pico_kib,
                       "avaliacoes_por_s": med.resultado.avaliacoes / med.tempo_s})
    for P in pops:
        cfg = ConfigGA(populacao=P, geracoes=geracoes, paciencia=None)
        med = medir(lambda: executar_ga(base_cx, base_k, cfg, criar_rng(seed, "bench-ga")), repeticoes=2)
        linhas.append({"variavel": "P (população)", "n": len(base_k), "P": P, "G": geracoes, "tempo_s": med.tempo_s,
                       "pico_kib": med.pico_kib, "avaliacoes_por_s": med.resultado.avaliacoes / med.tempo_s})
    return pd.DataFrame(linhas)


def greedy_escala(inst, fatores=(1, 10, 100), seed=0):
    linhas = []
    for f in fatores:
        ins = inst if f == 1 else ampliar(inst, f, seed)
        tab = construir_tabelas(ins)
        df = avaliar_politica(ins, tab, np.full(ins.n, nivel_referencia(ins)))
        med = medir(alocar_incentivos, ins, tab, df, repeticoes=2)
        linhas.append({"n_skus": ins.n, "lotes": 200, "tempo_s": med.tempo_s, "pico_kib": med.pico_kib})
    return pd.DataFrame(linhas)


def tabelas_escala(inst, fatores=(1, 10, 100), seed=0):
    linhas = []
    for f in fatores:
        ins = inst if f == 1 else ampliar(inst, f, seed)
        med = medir(construir_tabelas, ins, repeticoes=3)
        linhas.append({"n_skus": ins.n, "L": len(ins.niveis), "tempo_s": med.tempo_s, "pico_kib": med.pico_kib})
    return pd.DataFrame(linhas)


def otimizacao_antes_depois(inst, dom, seed, m=6):
    """Compara versão ingênua x otimizada: (1) Brute Force e (2) avaliação do fitness do GA."""
    sub = dom.subconjunto(escolher_subconjunto(dom, m, criar_rng(seed, "bench-bf")))
    c = dados_ingenuos(inst)
    perm = [list(sub.niveis_abs[r, :sub.k[r]]) for r in range(len(sub.k))]
    antes = medir(brute_force_ingenuo, c, sub.ativos.tolist(), perm, caixa_sku_ingenuo, repeticoes=1)
    depois = medir(brute_force, sub.caixa, sub.k, repeticoes=3)
    assert abs(antes.resultado[0] - depois.resultado.melhor_caixa) < 1e-6 * max(1, abs(depois.resultado.melhor_caixa))
    rng = criar_rng(seed, "bench-fit")
    pop = (rng.random((120, len(dom.k))) * dom.k).astype(np.int64)
    tab_l, pop_l = dom.caixa.tolist(), pop.tolist()
    reps = 300
    a2 = medir(lambda: [avaliar_populacao_ingenua(tab_l, pop_l) for _ in range(reps)], repeticoes=2)
    d2 = medir(lambda: [avaliar_populacao(dom.caixa, pop) for _ in range(reps)], repeticoes=3)
    return pd.DataFrame([
        {"alvo": f"Brute Force (m={m}, {depois.resultado.combinacoes} combinações)", "versao_antes": "itertools.product + recálculo em Python",
         "versao_depois": "tabelas pré-computadas + decodificação vetorizada em blocos", "tempo_antes_s": antes.tempo_s,
         "tempo_depois_s": depois.tempo_s, "speedup": antes.tempo_s / depois.tempo_s, "pico_antes_kib": antes.pico_kib, "pico_depois_kib": depois.pico_kib},
        {"alvo": f"Fitness do GA ({reps} avaliações de população 120x{len(dom.k)})", "versao_antes": "laços Python (soma gene a gene)",
         "versao_depois": "gather vetorizado numpy", "tempo_antes_s": a2.tempo_s, "tempo_depois_s": d2.tempo_s,
         "speedup": a2.tempo_s / d2.tempo_s, "pico_antes_kib": a2.pico_kib, "pico_depois_kib": d2.pico_kib}])
