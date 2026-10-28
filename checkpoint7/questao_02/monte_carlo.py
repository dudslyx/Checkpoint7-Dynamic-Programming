"""Etapa A (estimativa) e A' (validação) — Monte Carlo.

estimar_estatisticas(): amostra demanda/custo para TODOS os níveis de preço e devolve quantis (alimentam a DP).
simular(): simula 12 semanas (cascata de pagamentos) de uma política + plano promocional em cenários independentes.
Complexidade: estimativa O(N*n*L) (+ ordenação O(n*L*N log N) dos quantis); simulação O(N*W*n);
memória: estimativa O(N*n*L); simulação O(bloco*n + N*W) (cenários em blocos de 500; resultado independe do bloco).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from comum.aleatorio import criar_rng
from questao_02.modelo import InstanciaQ2

K_QUANTIS = 40
BLOCO = 500


@dataclass
class EstatisticasMC:
    n_cenarios: int
    dq: np.ndarray           # (n, L, K) quantis (pontos médios) da demanda semanal por SKU e nível
    media_dem: np.ndarray    # (n, L)
    desvio_dem: np.ndarray   # (n, L)
    frete_medio: np.ndarray  # (n,)
    caixa_media: np.ndarray  # (n, L) caixa semanal médio com estoque ilimitado
    caixa_p10: np.ndarray    # (n, L)


@dataclass
class PlanoPromo:
    gasto: np.ndarray        # (W, n) R$ por SKU e semana
    lift: np.ndarray         # (W, n) acréscimo relativo de demanda
    ganho_esperado: np.ndarray = field(default_factory=lambda: np.zeros(0))


def _choques(inst: InstanciaQ2, rng, B, logz=None):
    zd, zc, zf = rng.standard_normal((3, B, inst.n))
    xi = np.exp(inst.sig_d * zd - 0.5 * inst.sig_d ** 2)
    inc = inst.sig_c * zc - 0.5 * inst.sig_c ** 2
    if logz is None:
        zeta = inst.fator_conc * np.exp(inc)
    else:
        logz += inc
        zeta = inst.fator_conc * np.exp(logz)
    fr = inst.frete0 * np.exp(inst.sig_f * zf - 0.5 * inst.sig_f ** 2)
    return xi, zeta, fr


def estimar_estatisticas(inst: InstanciaQ2, n_cenarios: int, seed: int) -> EstatisticasMC:
    rng = criar_rng(seed, "mc-estimativa")
    xi, zeta, fr = _choques(inst, rng, n_cenarios)
    base = inst.demanda0 * zeta ** inst.elast * xi                                  # (N, n)
    D = base[:, :, None] * inst.niveis[None, None, :] ** (-inst.elast[:, None])    # (N, n, L) — números aleatórios comuns
    probs = (np.arange(K_QUANTIS) + 0.5) / K_QUANTIS
    dq = np.transpose(np.quantile(D, probs, axis=0), (1, 2, 0))
    preco_nl = inst.preco[:, None] * inst.niveis[None, :]
    margem = preco_nl[None] * (1 - inst.taxa[None, :, None]) - inst.custo[None, :, None] - fr[:, :, None]
    caixa = np.rint(D) * margem
    return EstatisticasMC(n_cenarios, dq, D.mean(0), D.std(0), fr.mean(0), caixa.mean(0), np.percentile(caixa, 10, axis=0))


# ---- políticas: funções (semana, estoque (B,n) int) -> nível (B,n) ----------------------------------------------
def politica_fixa(niveis_idx):
    arr = np.asarray(niveis_idx)
    return lambda w, est: np.broadcast_to(arr, est.shape)


def politica_caminho(caminho: np.ndarray):
    """Plano em malha aberta: preços por semana definidos antecipadamente (trajetória reconstruída da DP)."""
    return lambda w, est: np.broadcast_to(caminho[w], est.shape)


def politica_adaptativa(pol: np.ndarray):
    """Reprecificação semanal: consulta a tabela da DP pol[w, sku, estoque observado]."""
    idx = np.arange(pol.shape[1])
    return lambda w, est: pol[w][idx[None, :], est].astype(np.int64)


@dataclass
class ResultadoSim:
    caixa_sem: np.ndarray        # (N, W) caixa operacional líquido da semana
    receita: np.ndarray          # (N,)
    est_final_un: np.ndarray     # (N,)
    est_final_custo: np.ndarray  # (N,)
    traj_niv: np.ndarray | None = None   # (N, W, m) nível adotado nos SKUs acompanhados
    traj_est: np.ndarray | None = None   # (N, W+1, m) estoque


def simular(inst: InstanciaQ2, politica, promo: PlanoPromo | None, n_cenarios: int, seed: int, fluxo: str,
            persistente: bool = False, skus_traj=()) -> ResultadoSim:
    N, W, n = n_cenarios, inst.W, inst.n
    caixa = np.zeros((N, W)); receita = np.zeros(N); est_un = np.zeros(N); est_custo = np.zeros(N)
    sel = np.asarray(skus_traj, dtype=int)
    t_niv = np.zeros((N, W, len(sel)), dtype=np.int16) if len(sel) else None
    t_est = np.zeros((N, W + 1, len(sel))) if len(sel) else None
    gasto_sem = promo.gasto.sum(axis=1) if promo is not None else np.zeros(W)
    for j in range(-(-N // BLOCO)):
        s, e = j * BLOCO, min(N, (j + 1) * BLOCO)
        B = e - s
        rng = criar_rng(seed, f"{fluxo}-{j}")
        estoque = np.tile(inst.estoque0.astype(float), (B, 1))
        logz = np.zeros((B, n)) if persistente else None
        if len(sel): t_est[s:e, 0] = estoque[:, sel]
        for w in range(W):
            niv = np.asarray(politica(w, estoque.astype(np.int64)))
            r = inst.niveis[niv]
            xi, zeta, fr = _choques(inst, rng, B, logz)
            dem = inst.demanda0 * r ** (-inst.elast) * zeta ** inst.elast * xi
            if promo is not None:
                dem = dem * (1.0 + promo.lift[w])
            un = np.minimum(estoque, np.rint(dem))
            preco = inst.preco * r
            caixa[s:e, w] = (un * (preco * (1 - inst.taxa) - inst.custo - fr)).sum(1) - gasto_sem[w]
            receita[s:e] += (un * preco).sum(1)
            estoque -= un
            if len(sel):
                t_niv[s:e, w] = niv[:, sel]; t_est[s:e, w + 1] = estoque[:, sel]
        est_un[s:e] = estoque.sum(1)
        est_custo[s:e] = (estoque * inst.custo).sum(1)
    return ResultadoSim(caixa, receita, est_un, est_custo, t_niv, t_est)


def pagamentos(inst: InstanciaQ2, caixa_sem: np.ndarray):
    """Cascata: nas semanas de vencimento, paga se saldo >= dívida; senão registra falha e mantém o saldo."""
    N = caixa_sem.shape[0]
    saldo = np.full(N, inst.caixa_inicial)
    pagos = np.zeros((N, len(inst.semanas_pag)), dtype=bool)
    for w in range(inst.W):
        saldo = saldo + caixa_sem[:, w]
        if (w + 1) in inst.semanas_pag:
            j = inst.semanas_pag.index(w + 1)
            ok = saldo >= inst.dividas[j]
            saldo = np.where(ok, saldo - inst.dividas[j], saldo)
            pagos[:, j] = ok
    return saldo, pagos


def resumir(inst: InstanciaQ2, res: ResultadoSim, pct_risco: float = 10.0) -> dict:
    _, pagos = pagamentos(inst, res.caixa_sem)
    N = len(res.receita)
    acum = inst.caixa_inicial + res.caixa_sem.sum(1)
    liquido = acum - inst.dividas.sum()
    p_r = float(np.percentile(liquido, pct_risco))
    p_todas = float(pagos.all(1).mean())
    pp = {int(s): float(pagos[:, j].mean()) for j, s in enumerate(inst.semanas_pag)}
    return {"n_cenarios": N, "p_pagamento": pp, "p_todas": p_todas, "cenarios_falha": int((~pagos.all(1)).sum()),
            "falhas_por_prazo": {int(s): int((~pagos[:, j]).sum()) for j, s in enumerate(inst.semanas_pag)},
            "caixa_acumulado_medio": float(acum.mean()), "saldo_liquido_medio": float(liquido.mean()),
            "saldo_liquido_desvio": float(liquido.std()), "percentil_risco": pct_risco, "saldo_liquido_percentil": p_r,
            "cvar": float(liquido[liquido <= p_r].mean()), "margem_pct": float(100 * res.caixa_sem.sum() / res.receita.sum()),
            "estoque_final_un": float(res.est_final_un.mean()),
            "estoque_final_pct": float(100 * res.est_final_un.mean() / inst.estoque0.sum()),
            "estoque_final_custo": float(res.est_final_custo.mean()),
            "risco_ok": bool((1 - p_todas) <= inst.limite_risco and all(1 - v <= inst.limite_risco for v in pp.values())),
            "liquido": liquido}
