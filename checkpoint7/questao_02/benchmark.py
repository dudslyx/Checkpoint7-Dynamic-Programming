"""Medições reais (perf_counter + tracemalloc) da Questão 2: três tamanhos de entrada e de nº de cenários."""
from __future__ import annotations

import pandas as pd

from comum.metricas import medir
from questao_02.dynamic_programming import resolver_dp, reconstruir_caminho
from questao_02.greedy import alocar_promocoes
from questao_02.modelo import ampliar
from questao_02.monte_carlo import estimar_estatisticas, politica_adaptativa, simular


def escala_instancia(inst, seed, fatores=(1, 4, 16), n_est=2000, n_val=1000):
    linhas = []
    for f in fatores:
        ins = inst if f == 1 else ampliar(inst, f)
        m_est = medir(estimar_estatisticas, ins, n_est, seed)
        est = m_est.resultado
        m_dp = medir(resolver_dp, ins, est, 0.0)
        sol = m_dp.resultado
        cam = reconstruir_caminho(ins, est, sol)
        m_gr = medir(alocar_promocoes, ins, est, cam, sol)
        plano = m_gr.resultado
        m_mc = medir(simular, ins, politica_adaptativa(sol.politica), plano, n_val, seed, "bench")
        for etapa, m in [("Monte Carlo — estimativa", m_est), ("Programação Dinâmica", m_dp), ("Greedy semanal", m_gr), ("Monte Carlo — simulação", m_mc)]:
            linhas.append({"n_skus": ins.n, "etapa": etapa, "tempo_s": m.tempo_s, "pico_kib": m.pico_kib,
                           "cenarios": n_est if "estimativa" in etapa else (n_val if "simulação" in etapa else None)})
    return pd.DataFrame(linhas)


def escala_cenarios(inst, seed, Ns=(1000, 2000, 5000, 10000)):
    est = estimar_estatisticas(inst, 2000, seed)
    sol = resolver_dp(inst, est, 0.0)
    pol = politica_adaptativa(sol.politica)
    linhas = []
    for N in Ns:
        m = medir(simular, inst, pol, None, N, seed, "bench-n")
        linhas.append({"n_cenarios": N, "tempo_s": m.tempo_s, "pico_kib": m.pico_kib, "cenarios_por_s": N / m.tempo_s})
    return pd.DataFrame(linhas)
