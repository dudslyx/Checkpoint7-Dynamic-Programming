"""Modelagem da Questão 2 (Plano de Sobrevivência em 12 Semanas).

Por SKU i, semana w, nível de preço l (multiplicador sobre preco_atual), choques aleatórios (lognormais de média 1):
    xi   ~ LogNormal(-s_d^2/2, s_d)   demanda do consumidor
    zeta ~ LogNormal(-s_c^2/2, s_c)   preço do concorrente relativo à base (x fator_concorrente do cenário-surpresa)
    phi  ~ LogNormal(-s_f^2/2, s_f)   custo logístico relativo à base
    demanda    = demanda_base_semanal * nivel^(-elasticidade) * zeta^elasticidade * xi * (1 + lift_promocional)
    unidades   = min(estoque, round(demanda))
    caixa_sem  = sum_i unidades * (preço*(1-taxa) - custo - custo_logistico_base*phi)  - gasto promocional
Pagamentos nas semanas 4, 8 e 12: paga se o saldo acumulado >= dívida (cascata); senão, falha no prazo.
"""
from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import pandas as pd

from comum.cenarios import Ajustes

CAMPOS = ["sku", "categoria", "custo", "preco", "estoque0", "demanda0", "elast", "taxa", "frete0", "preco_conc0"]


@dataclass
class InstanciaQ2:
    sku: np.ndarray
    categoria: np.ndarray
    custo: np.ndarray
    preco: np.ndarray
    estoque0: np.ndarray       # int64
    demanda0: np.ndarray
    elast: np.ndarray
    taxa: np.ndarray
    frete0: np.ndarray
    preco_conc0: np.ndarray
    niveis: np.ndarray
    W: int
    semanas_pag: tuple
    dividas: np.ndarray
    caixa_inicial: float
    sig_d: float
    sig_c: float
    sig_f: float
    orcamento_sem: float
    n_sim: int = 2000
    limite_risco: float = 0.10
    fator_conc: float = 1.0

    @property
    def n(self) -> int: return len(self.sku)
    @property
    def L(self) -> int: return len(self.niveis)
    @property
    def ref(self) -> int: return int(np.argmin(np.abs(self.niveis - 1.0)))


def construir_instancia(df: pd.DataFrame, cfg: dict, ajustes: Ajustes = Ajustes()) -> InstanciaQ2:
    df = df[~df["sku"].isin(ajustes.skus_indisponiveis)].reset_index(drop=True)
    if df.empty:
        raise ValueError("Nenhum SKU restante após remover os indisponíveis.")
    pares = sorted((int(k.split("_")[-1]), float(v)) for k, v in cfg["dividas"].items())
    v = cfg["variabilidade"]
    return InstanciaQ2(
        sku=df["sku"].to_numpy(), categoria=df["categoria"].to_numpy(), custo=df["custo_unitario"].to_numpy(float),
        preco=df["preco_atual"].to_numpy(float), estoque0=df["estoque_inicial"].to_numpy(np.int64),
        demanda0=df["demanda_base_semanal"].to_numpy(float), elast=df["elasticidade"].to_numpy(float),
        taxa=df["taxa_marketplace"].to_numpy(float), frete0=df["custo_logistico_base"].to_numpy(float) * ajustes.custo_logistico_fator,
        preco_conc0=df["preco_concorrente_base"].to_numpy(float), niveis=np.array(cfg["niveis_preco_relativos"], dtype=float),
        W=int(cfg["numero_semanas"]), semanas_pag=tuple(s for s, _ in pares),
        dividas=np.array([d for _, d in pares]) * ajustes.divida_fator, caixa_inicial=float(cfg["caixa_inicial"]),
        sig_d=float(v["sigma_demanda"]), sig_c=float(v["sigma_preco_concorrente"]), sig_f=float(v["sigma_custo_logistico"]),
        orcamento_sem=float(cfg["orcamento_promocional_semanal"]) * ajustes.orcamento_fator,
        n_sim=int(cfg.get("n_simulacoes_recomendadas", 2000)), limite_risco=float(cfg.get("limite_risco_falha", 0.10)),
        fator_conc=1.0 + ajustes.concorrente_delta)


def ampliar(inst: InstanciaQ2, fator: int) -> InstanciaQ2:
    """Instância sintética maior (benchmarks): replica SKUs e escala dívidas/caixa/orçamento proporcionalmente."""
    novo = {c: np.tile(getattr(inst, c), fator) for c in CAMPOS}
    novo["sku"] = np.array([f"S{j:05d}" for j in range(inst.n * fator)])
    return replace(inst, **novo, dividas=inst.dividas * fator, caixa_inicial=inst.caixa_inicial * fator,
                   orcamento_sem=inst.orcamento_sem * fator)


def peso_semana(W: int, semanas_pag: tuple, theta: float) -> np.ndarray:
    """Peso de urgência do caixa da semana w: 1 + theta*(d_w - 1)/(D - 1), d_w = nº de prazos ainda não vencidos (>= w).
    theta=0 => objetivo = caixa esperado total; theta>0 => antecipa caixa para cumprir os prazos de curto prazo."""
    D = len(semanas_pag)
    d = np.array([max(1, sum(1 for s in semanas_pag if s >= w + 1)) for w in range(W)], dtype=float)
    return np.ones(W) + (theta * (d - 1) / (D - 1) if D > 1 else np.zeros(W))
