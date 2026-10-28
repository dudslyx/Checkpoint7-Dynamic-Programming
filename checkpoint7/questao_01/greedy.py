"""Etapa C — Greedy: distribui o orçamento de incentivo APÓS a política de preços do GA.

Entrada: política do GA (preço por SKU). Regra: investir sempre o próximo lote no SKU de maior
benefício marginal por real investido = margem_un * delta_unidades / lote (parando quando < R$ 1,00 por R$ 1,00).
Complexidade: O((n + L) log n) tempo, O(n) memória (L = nº de lotes).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from comum.incentivos import PARAMS_PADRAO, ParamsIncentivo, alocar_guloso, escala_saturacao, fator_lift, lift_maximo
from questao_01.modelo import InstanciaQ1, Tabelas


@dataclass
class ResultadoGreedy:
    gasto_por_sku: np.ndarray
    gasto_total: float
    lotes: int
    detalhe: pd.DataFrame      # antes/depois por SKU


def alocar_incentivos(inst: InstanciaQ1, tab: Tabelas, df_politica: pd.DataFrame,
                      params: ParamsIncentivo = PARAMS_PADRAO) -> ResultadoGreedy:
    ativo = (df_politica["status"] == "ativo").to_numpy()
    dem = df_politica["demanda"].to_numpy()
    un0 = df_politica["unidades"].to_numpy()
    m = df_politica["margem_un"].to_numpy()
    lmax = lift_maximo(inst.elasticidade, params)
    xs = np.where(ativo, escala_saturacao(df_politica["receita"].to_numpy(), params), 0.0)
    est = inst.estoque

    def unidades_fn(i, x):
        if not ativo[i]:
            return 0.0
        return float(min(est[i], dem[i] * fator_lift(x, lmax[i], xs[i])))

    lote = inst.orcamento / params.n_lotes if params.n_lotes else 0.0
    x, gasto, lotes = alocar_guloso(inst.n, inst.orcamento, lote, lambda i, xi: m[i] * unidades_fn(i, xi))
    un1 = np.array([unidades_fn(i, x[i]) for i in range(inst.n)])
    ganho_bruto = m * (un1 - un0)
    det = df_politica[["sku", "categoria", "preco", "unidades", "caixa", "estoque_restante"]].copy()
    det = det.rename(columns={"unidades": "unidades_antes", "caixa": "caixa_antes", "estoque_restante": "estoque_rest_antes"})
    det["incentivo_rs"] = x
    det["unidades_depois"] = un1
    det["caixa_depois"] = df_politica["caixa"].to_numpy() + ganho_bruto - x
    det["estoque_rest_depois"] = inst.estoque - un1
    det["beneficio_por_real"] = np.where(x > 0, ganho_bruto / np.where(x > 0, x, 1.0), 0.0)
    return ResultadoGreedy(x, float(gasto), lotes, det)
