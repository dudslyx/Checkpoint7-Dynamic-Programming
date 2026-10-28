"""Etapa B — Programação Dinâmica (bottom-up) para a reprecificação semanal.

Decomposição: SKUs só interagem pelo caixa total e pelo orçamento promocional; para cada SKU resolve-se uma DP exata.
  Estado      : (w, s)  semana w e estoque restante s do SKU (0..estoque inicial) — inteiro, sem aproximação.
  Decisão     : nível de preço l da semana.
  Transição   : s' = s - min(s, d_k), d_k = k-ésimo quantil da demanda (Monte Carlo), k = 1..K equiprováveis.
  Objetivo    : V(w,s) = max_l  (1/K) * sum_k [ peso_w * margem_l * min(s, d_k) + V(w+1, s - min(s, d_k)) ]
  Terminal    : V(W, s) = 0 (estoque que sobra não gera caixa no horizonte).
Reutilização de subproblemas: cada V(w+1, s') é lido por TODOS os (s, l, k) que levam a s' — O(L*K) leituras por estado,
em vez de enumerar os L^W caminhos de preço. A trajetória ótima é reconstruída seguindo a política a partir de s0.
Complexidade: tempo O(n*W*L*S*K); memória O(n*W*S) (tabela de política int8) + O(S*K*L) temporária por SKU.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from questao_02.modelo import InstanciaQ2, peso_semana
from questao_02.monte_carlo import EstatisticasMC


@dataclass
class SolucaoDP:
    politica: np.ndarray      # (W, n, Smax+1) int8
    valor0: np.ndarray        # (n,) V(0, s0)
    theta: float
    pesos: np.ndarray
    estados: int              # subproblemas distintos (w, s) resolvidos
    leituras: int             # leituras de V(w+1, .) (reutilizações)
    caminhos_ingenuos: float  # L^W: nº de trajetórias de preço que a enumeração testaria
    valor: np.ndarray = None  # (W+1, n, Smax+1) V(w, s) — valor futuro do estoque (usado pelo Greedy)


def resolver_dp(inst: InstanciaQ2, est: EstatisticasMC, theta: float = 0.0) -> SolucaoDP:
    W, n, L = inst.W, inst.n, inst.L
    K = est.dq.shape[2]
    smax = int(inst.estoque0.max())
    pesos = peso_semana(W, inst.semanas_pag, theta)
    pol = np.full((W, n, smax + 1), inst.ref, dtype=np.int8)
    Vtab = np.zeros((W + 1, n, smax + 1))
    ordem = sorted(range(L), key=lambda l: abs(l - inst.ref))      # desempate: nível mais próximo da referência
    valor0 = np.zeros(n)
    estados = 0
    for i in range(n):
        S = int(inst.estoque0[i])
        s = np.arange(S + 1)
        dq = np.rint(est.dq[i])                                      # (L, K) unidades demandadas
        margem = inst.preco[i] * inst.niveis * (1 - inst.taxa[i]) - inst.custo[i] - est.frete_medio[i]
        vendas = [np.minimum(s[:, None], dq[l][None, :]).astype(np.int64) for l in range(L)]
        prox = [s[:, None] - v for v in vendas]
        ganho = [margem[l] * v.mean(axis=1) for l, v in enumerate(vendas)]
        V = np.zeros(S + 1)
        for w in range(W - 1, -1, -1):
            melhor, arg = None, None
            for l in ordem:
                val = pesos[w] * ganho[l] + V[prox[l]].mean(axis=1)
                if melhor is None:
                    melhor, arg = val, np.full(S + 1, l)
                else:
                    m = val > melhor + 1e-9
                    melhor, arg = np.where(m, val, melhor), np.where(m, l, arg)
            V = melhor
            pol[w, i, :S + 1] = arg
            Vtab[w, i, :S + 1] = V
        valor0[i] = V[S]
        estados += W * (S + 1)
    return SolucaoDP(pol, valor0, theta, pesos, estados, estados * L * K, float(L) ** W, Vtab)


@dataclass
class Caminho:
    niveis: np.ndarray       # (W, n) nível de preço do caminho ótimo esperado
    estoque_ini: np.ndarray  # (W, n) estoque esperado no início da semana
    vendas_esp: np.ndarray   # (W, n)
    caixa_esp: np.ndarray    # (W, n)


def reconstruir_caminho(inst: InstanciaQ2, est: EstatisticasMC, sol: SolucaoDP) -> Caminho:
    """Segue a política ótima a partir do estoque inicial, avançando o estoque pela venda esperada."""
    W, n = inst.W, inst.n
    niv = np.zeros((W, n), dtype=np.int64); e_ini = np.zeros((W, n), dtype=np.int64)
    v_esp = np.zeros((W, n)); c_esp = np.zeros((W, n))
    s = inst.estoque0.copy()
    for w in range(W):
        l = sol.politica[w, np.arange(n), s].astype(np.int64)
        dq = np.rint(est.dq[np.arange(n), l, :])
        u = np.minimum(s[:, None], dq).mean(axis=1)
        margem = inst.preco * inst.niveis[l] * (1 - inst.taxa) - inst.custo - est.frete_medio
        niv[w], e_ini[w], v_esp[w], c_esp[w] = l, s, u, u * margem
        s = np.maximum(0, np.rint(s - u)).astype(np.int64)
    return Caminho(niv, e_ini, v_esp, c_esp)
