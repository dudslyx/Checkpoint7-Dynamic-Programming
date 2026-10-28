"""Modelo de resposta a incentivos comerciais (cupom/cashback/frete subsidiado/mídia) e Greedy genérico.

Premissa de modelagem (os dados oficiais não trazem curva de resposta; regra justificada em docs/decisoes.md):
    demanda_com_incentivo(x) = demanda * (1 + lift_max * (1 - exp(-x / x_sat)))        [retornos decrescentes]
    lift_max = min(LIFT_TETO, LIFT_POR_ELASTICIDADE * elasticidade)   (itens mais sensíveis a preço respondem mais)
    x_sat    = SATURACAO_PCT_RECEITA * receita bruta de referência do SKU
    unidades = min(estoque, demanda_com_incentivo(x))         -> o incentivo só ajuda enquanto há estoque.
O custo do incentivo (x, em R$) é descontado do caixa.
"""
from __future__ import annotations

import heapq
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ParamsIncentivo:
    lift_por_elasticidade: float = 0.15
    lift_teto: float = 0.40
    saturacao_pct_receita: float = 0.04
    n_lotes: int = 200          # granularidade do Greedy: orçamento dividido em n_lotes partes iguais


PARAMS_PADRAO = ParamsIncentivo()


def lift_maximo(elasticidade, p: ParamsIncentivo = PARAMS_PADRAO):
    return np.minimum(p.lift_teto, p.lift_por_elasticidade * np.asarray(elasticidade, dtype=float))


def escala_saturacao(receita_ref, p: ParamsIncentivo = PARAMS_PADRAO):
    return p.saturacao_pct_receita * np.asarray(receita_ref, dtype=float)


def fator_lift(x, lmax, xs):
    """Multiplicador de demanda (1 + lift) para gasto x; vetorizado. Sem escala (xs<=0) -> 1."""
    x, lmax, xs = np.broadcast_arrays(np.asarray(x, float), np.asarray(lmax, float), np.asarray(xs, float))
    seguro = np.where(xs > 0, xs, 1.0)
    return 1.0 + np.where(xs > 0, lmax * (1.0 - np.exp(-x / seguro)), 0.0)


def alocar_guloso(n: int, orcamento: float, lote: float, valor_fn, ganho_minimo: float = 0.0):
    """Greedy por benefício marginal por real investido (heap de máximo).

    valor_fn(i, x) -> benefício bruto esperado (R$) do item i com gasto acumulado x (côncavo em x). Na Q1 é
                      margem_un * unidades(x); na Q2 inclui o valor futuro do estoque dado pela DP.
    A cada passo investe um lote no item cujo (valor(x+lote) - valor(x) - lote) / lote é o maior;
    para quando o melhor ganho líquido <= ganho_minimo ou o orçamento acaba.
    Complexidade: O((n + L) log n), L = nº de lotes; memória O(n).
    Retorna (gasto por item, gasto total, lotes usados).
    """
    x = np.zeros(n)
    if orcamento <= 0 or lote <= 0:
        return x, 0.0, 0

    def ganho(i):
        return valor_fn(i, x[i] + lote) - valor_fn(i, x[i]) - lote

    heap = [(-ganho(i) / lote, i) for i in range(n)]
    heapq.heapify(heap)
    gasto, lotes = 0.0, 0
    while heap and gasto + lote <= orcamento + 1e-9:
        neg, i = heapq.heappop(heap)
        if -neg <= ganho_minimo:
            break
        x[i] += lote
        gasto += lote
        lotes += 1
        heapq.heappush(heap, (-ganho(i) / lote, i))
    return x, gasto, lotes
