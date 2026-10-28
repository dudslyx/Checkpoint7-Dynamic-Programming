"""Modelagem da Questão 1 (Operação Caixa 90): instância, tabelas de avaliação e métricas de política.

Modelo (determinístico, valores esperados):
    preço(i,l)      = preco_atual_i * nivel_l
    demanda(i,l)    = demanda_base_90d_i * (preço/preco_atual)^(-elasticidade_i)
    unidades(i,l)   = min(estoque_i, demanda(i,l))
    margem_un(i,l)  = preço*(1 - taxa_marketplace) - custo_unitario - custo_logistico
    caixa(i,l)      = unidades * margem_un         (mesma definição de 'caixa base' do gerador oficial)
    permitido(i,l)  = margem_un/preço >= margem_minima_percentual   (piso de margem por SKU)
"""
from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import pandas as pd

from comum.cenarios import Ajustes

CAMPOS = ["sku", "categoria", "custo", "preco", "estoque", "demanda", "elasticidade", "taxa", "frete", "idade"]


@dataclass
class InstanciaQ1:
    sku: np.ndarray
    categoria: np.ndarray
    custo: np.ndarray
    preco: np.ndarray
    estoque: np.ndarray
    demanda: np.ndarray
    elasticidade: np.ndarray
    taxa: np.ndarray
    frete: np.ndarray
    idade: np.ndarray
    niveis: np.ndarray
    margem_min: float
    meta: float
    divida: float
    orcamento: float
    horizonte: int = 90

    @property
    def n(self) -> int:
        return len(self.sku)


def construir_instancia(df: pd.DataFrame, cfg: dict, ajustes: Ajustes = Ajustes()) -> InstanciaQ1:
    df = df[~df["sku"].isin(ajustes.skus_indisponiveis)].reset_index(drop=True)
    if df.empty:
        raise ValueError("Nenhum SKU restante após remover os indisponíveis.")
    return InstanciaQ1(
        sku=df["sku"].to_numpy(), categoria=df["categoria"].to_numpy(),
        custo=df["custo_unitario"].to_numpy(float), preco=df["preco_atual"].to_numpy(float),
        estoque=df["estoque"].to_numpy(float), demanda=df["demanda_base_90d"].to_numpy(float),
        elasticidade=df["elasticidade"].to_numpy(float), taxa=df["taxa_marketplace"].to_numpy(float),
        frete=df["custo_logistico"].to_numpy(float) * ajustes.custo_logistico_fator,
        idade=df["idade_estoque_dias"].to_numpy(float),
        niveis=np.array(cfg["niveis_preco_relativos"], dtype=float),
        margem_min=float(cfg["margem_minima_percentual"]),
        meta=float(cfg["meta_caixa_90d"]) * ajustes.meta_fator,
        divida=float(cfg["divida_total_90d"]),
        orcamento=float(cfg["orcamento_promocional"]) * ajustes.orcamento_fator,
        horizonte=int(cfg.get("horizonte_dias", 90)))


def ampliar(inst: InstanciaQ1, fator: int, seed: int = 0) -> InstanciaQ1:
    """Instância sintética maior (benchmarks): replica o catálogo com ruído lognormal na demanda."""
    rng = np.random.default_rng(seed)
    novo = {c: np.tile(getattr(inst, c), fator) for c in CAMPOS}
    novo["sku"] = np.array([f"S{j:05d}" for j in range(inst.n * fator)])
    novo["demanda"] = novo["demanda"] * rng.lognormal(0, 0.1, inst.n * fator)
    return replace(inst, **novo, orcamento=inst.orcamento * fator, meta=inst.meta * fator, divida=inst.divida * fator)


@dataclass
class Tabelas:
    precos: np.ndarray
    demanda: np.ndarray
    unidades: np.ndarray
    margem_un: np.ndarray
    margem_pct: np.ndarray
    caixa: np.ndarray
    receita: np.ndarray
    permitido: np.ndarray


def construir_tabelas(inst: InstanciaQ1) -> Tabelas:
    """Pré-computa todas as combinações (SKU, nível): O(n*L) tempo e memória. Evita recalcular no GA/BF."""
    precos = np.round(inst.preco[:, None] * inst.niveis[None, :], 2)
    razao = precos / inst.preco[:, None]
    demanda = inst.demanda[:, None] * razao ** (-inst.elasticidade[:, None])
    unidades = np.minimum(inst.estoque[:, None], demanda)
    margem_un = precos * (1 - inst.taxa[:, None]) - inst.custo[:, None] - inst.frete[:, None]
    margem_pct = margem_un / precos
    return Tabelas(precos, demanda, unidades, margem_un, margem_pct, unidades * margem_un, unidades * precos,
                   margem_pct >= inst.margem_min - 1e-12)


@dataclass
class Dominio:
    """Espaço de busca compacto: para cada SKU ativo, a lista de níveis permitidos pelo piso de margem."""
    ativos: np.ndarray      # índices dos SKUs que possuem ao menos um preço permitido
    k: np.ndarray           # nº de preços permitidos por SKU ativo
    niveis_abs: np.ndarray  # (n_ativos, Kmax) índice do nível em inst.niveis (-1 = preenchimento)
    caixa: np.ndarray       # (n_ativos, Kmax) caixa de cada preço permitido (-1e18 = preenchimento)

    def subconjunto(self, linhas) -> "Dominio":
        linhas = np.asarray(linhas)
        return Dominio(self.ativos[linhas], self.k[linhas], self.niveis_abs[linhas], self.caixa[linhas])


def construir_dominio(tab: Tabelas) -> Dominio:
    ativos = np.where(tab.permitido.any(axis=1))[0]
    k = tab.permitido[ativos].sum(axis=1).astype(np.int64)
    kmax = int(k.max()) if len(k) else 0
    niv = np.full((len(ativos), kmax), -1, dtype=np.int64)
    cx = np.full((len(ativos), kmax), -1e18)
    for r, i in enumerate(ativos):
        idx = np.where(tab.permitido[i])[0]
        niv[r, :len(idx)] = idx
        cx[r, :len(idx)] = tab.caixa[i, idx]
    return Dominio(ativos, k, niv, cx)


def escolher_subconjunto(dom: Dominio, m: int, rng: np.random.Generator) -> np.ndarray:
    """m SKUs com maior nº de preços permitidos (instância reduzida mais difícil); desempate por sorteio com seed."""
    m = min(m, len(dom.ativos))
    desempate = rng.random(len(dom.ativos))
    ordem = np.lexsort((desempate, -dom.k))
    return np.sort(ordem[:m])


def genes_para_niveis(dom: Dominio, genes: np.ndarray, n: int) -> np.ndarray:
    """Converte genes (posição na lista de permitidos) em índice de nível por SKU; -1 = SKU suspenso."""
    niveis = np.full(n, -1, dtype=np.int64)
    niveis[dom.ativos] = dom.niveis_abs[np.arange(len(dom.ativos)), genes]
    return niveis


def nivel_referencia(inst: InstanciaQ1) -> int:
    return int(np.argmin(np.abs(inst.niveis - 1.0)))


def avaliar_politica(inst: InstanciaQ1, tab: Tabelas, niveis: np.ndarray) -> pd.DataFrame:
    """Detalhe por SKU de uma política (niveis[i] = índice do nível; -1 = SKU suspenso, sem vendas)."""
    ativo = niveis >= 0
    j = np.where(ativo, niveis, 0)
    r = np.arange(inst.n)
    un = np.where(ativo, tab.unidades[r, j], 0.0)
    caixa = np.where(ativo, tab.caixa[r, j], 0.0)
    preco = np.where(ativo, tab.precos[r, j], inst.preco)
    return pd.DataFrame({
        "sku": inst.sku, "categoria": inst.categoria, "status": np.where(ativo, "ativo", "suspenso (nenhum preço respeita o piso de margem)"),
        "nivel": np.where(ativo, inst.niveis[j], np.nan), "preco": preco,
        "demanda": np.where(ativo, tab.demanda[r, j], 0.0), "unidades": un,
        "margem_un": np.where(ativo, tab.margem_un[r, j], 0.0), "margem_pct": np.where(ativo, tab.margem_pct[r, j], 0.0),
        "caixa": caixa, "receita": un * preco, "estoque": inst.estoque, "estoque_restante": inst.estoque - un,
        "idade_estoque_dias": inst.idade, "custo_unitario": inst.custo})


def totais(df: pd.DataFrame, gasto_incentivo: float = 0.0) -> dict:
    caixa = float(df["caixa"].sum() - gasto_incentivo)
    receita = float(df["receita"].sum())
    resta = df["estoque_restante"].to_numpy()
    return {"caixa": caixa, "receita_bruta": receita, "margem_pct": caixa / receita if receita else 0.0,
            "unidades_vendidas": float(df["unidades"].sum()), "estoque_remanescente_un": float(resta.sum()),
            "estoque_remanescente_custo": float((resta * df["custo_unitario"].to_numpy()).sum()),
            "estoque_antigo_180d_custo": float((resta * df["custo_unitario"].to_numpy() * (df["idade_estoque_dias"].to_numpy() > 180)).sum()),
            "gasto_incentivo": gasto_incentivo}


def caixa_sku_ingenuo(c, i: int, l: int) -> float:
    """Versão 'antes' (sem tabelas): recalcula tudo do zero em Python puro. Usada só na comparação de otimização."""
    p = round(c["preco"][i] * c["niveis"][l], 2)
    d = c["demanda"][i] * (p / c["preco"][i]) ** (-c["elast"][i])
    u = min(c["estoque"][i], d)
    return u * (p * (1 - c["taxa"][i]) - c["custo"][i] - c["frete"][i])


def dados_ingenuos(inst: InstanciaQ1) -> dict:
    return {"preco": inst.preco.tolist(), "niveis": inst.niveis.tolist(), "demanda": inst.demanda.tolist(),
            "elast": inst.elasticidade.tolist(), "estoque": inst.estoque.tolist(), "taxa": inst.taxa.tolist(),
            "custo": inst.custo.tolist(), "frete": inst.frete.tolist()}
