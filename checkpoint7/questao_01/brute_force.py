"""Etapa A — Brute Force: enumera TODAS as combinações de preços permitidos de um subconjunto pequeno.

Serve de benchmark ótimo para validar o Algoritmo Genético (gap percentual).
Complexidade: tempo O(m * K^m) (K = preços permitidos por SKU, m = nº de SKUs); memória O(m*K + bloco).
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass

import numpy as np

LIMITE_COMBINACOES = 600_000_000


class LimiteBruteForce(Exception):
    """Espaço de busca grande demais para enumeração exaustiva."""


@dataclass
class ResultadoBF:
    melhor_caixa: float
    melhores_genes: np.ndarray
    combinacoes: int


def contar_combinacoes(k) -> int:
    total = 1
    for v in k:
        total *= int(v)
    return total


def brute_force(caixa: np.ndarray, k: np.ndarray, bloco: int = 1 << 20) -> ResultadoBF:
    """Versão OTIMIZADA: tabelas pré-computadas + decodificação vetorizada (base mista) em blocos.

    caixa[i, j] = caixa do SKU i com o j-ésimo preço permitido; k[i] = nº de preços permitidos.
    A combinação de índice `c` (0 <= c < prod(k)) é decodificada em dígitos de base mista.
    """
    k = np.asarray(k, dtype=np.int64)
    m = len(k)
    total = contar_combinacoes(k)
    if total > LIMITE_COMBINACOES:
        raise LimiteBruteForce(f"{total:.3e} combinações excedem o limite ({LIMITE_COMBINACOES:.1e}); use GA/DP/heurísticas.")
    if m == 0:
        return ResultadoBF(0.0, np.zeros(0, dtype=np.int64), 1)
    passos = np.ones(m, dtype=np.int64)
    for i in range(m - 2, -1, -1):
        passos[i] = passos[i + 1] * k[i + 1]
    linhas = np.arange(m)
    melhor, melhor_idx = -np.inf, 0
    for ini in range(0, total, bloco):
        idx = np.arange(ini, min(ini + bloco, total), dtype=np.int64)
        valor = np.zeros(len(idx))
        for i in range(m):
            valor += caixa[i][(idx // passos[i]) % k[i]]
        j = int(np.argmax(valor))
        if valor[j] > melhor:                      # '>' estrito: desempate pela 1ª combinação (determinístico)
            melhor, melhor_idx = float(valor[j]), int(idx[j])
    genes = (melhor_idx // passos) % k
    return ResultadoBF(melhor, genes.astype(np.int64), total)


def brute_force_ingenuo(c: dict, sku_idx, niveis_perm, caixa_fn) -> tuple[float, tuple]:
    """Versão 'ANTES' (sem otimização): itertools.product + recálculo do caixa em Python puro a cada combinação."""
    melhor, melhor_comb = -float("inf"), None
    for comb in itertools.product(*niveis_perm):
        v = 0.0
        for i, l in zip(sku_idx, comb):
            v += caixa_fn(c, i, l)
        if v > melhor:
            melhor, melhor_comb = v, comb
    return melhor, melhor_comb
