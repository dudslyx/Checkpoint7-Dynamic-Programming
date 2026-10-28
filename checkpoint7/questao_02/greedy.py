"""Etapa C — Greedy semanal: aloca o recurso promocional escasso (destaques/mídia/frete) guiado pelo plano da DP.

Para cada semana w: usa o preço e o estoque esperado do caminho da DP, calcula o ganho esperado de caixa por lote e
investe sempre no SKU de maior benefício marginal por real (comum.incentivos.alocar_guloso).
Complexidade: O(W * (n + L) * K * log n) tempo (K = quantis usados em cada avaliação); memória O(W*n).
"""
from __future__ import annotations

import numpy as np

from comum.incentivos import PARAMS_PADRAO, ParamsIncentivo, alocar_guloso, fator_lift, lift_maximo
from questao_02.dynamic_programming import Caminho, SolucaoDP
from questao_02.modelo import InstanciaQ2
from questao_02.monte_carlo import EstatisticasMC, PlanoPromo

LOTES_POR_SEMANA = 40


def alocar_promocoes(inst: InstanciaQ2, est: EstatisticasMC, cam: Caminho, sol: SolucaoDP, params: ParamsIncentivo = PARAMS_PADRAO,
                     lotes_semana: int = LOTES_POR_SEMANA) -> PlanoPromo:
    W, n = inst.W, inst.n
    gasto, lift, ganho_sem = np.zeros((W, n)), np.zeros((W, n)), np.zeros(W)
    lmax = lift_maximo(inst.elast, params)
    idx = np.arange(n)
    for w in range(W):
        s = cam.estoque_ini[w].astype(float)
        dq = np.rint(est.dq[idx, cam.niveis[w], :])
        preco = inst.preco * inst.niveis[cam.niveis[w]]
        margem = preco * (1 - inst.taxa) - inst.custo - est.frete_medio
        u0 = np.minimum(s[:, None], dq).mean(axis=1)
        xs = params.saturacao_pct_receita * u0 * preco

        V1 = sol.valor[w + 1]
        peso = sol.pesos[w]

        def valor(i, x, s=s, dq=dq, xs=xs, margem=margem, V1=V1, peso=peso):
            """Benefício bruto esperado (R$) = caixa da semana + valor futuro do estoque restante (V da DP)."""
            vend = np.minimum(s[i], dq[i] * fator_lift(x, lmax[i], xs[i]))
            grade = np.arange(int(inst.estoque0[i]) + 1)
            return float(peso * margem[i] * vend.mean() + np.interp(s[i] - vend, grade, V1[i, :len(grade)]).mean())

        x, g, _ = alocar_guloso(n, inst.orcamento_sem, inst.orcamento_sem / lotes_semana, valor)
        gasto[w] = x
        lift[w] = fator_lift(x, lmax, xs) - 1.0
        ganho_sem[w] = float(sum(valor(i, x[i]) - valor(i, 0.0) for i in range(n)) - g)
    return PlanoPromo(gasto, lift, ganho_sem)
