"""Carga, validação e verificação de integridade (SHA-256) dos arquivos oficiais de entrada."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

RAIZ = Path(__file__).resolve().parents[1]
PASTA_DADOS = RAIZ / "data"

COLUNAS_Q1 = ["sku", "categoria", "custo_unitario", "preco_atual", "estoque", "demanda_base_90d",
              "elasticidade", "taxa_marketplace", "custo_logistico", "idade_estoque_dias"]
COLUNAS_Q2 = ["sku", "categoria", "custo_unitario", "preco_atual", "estoque_inicial", "demanda_base_semanal",
              "elasticidade", "taxa_marketplace", "custo_logistico_base", "preco_concorrente_base"]


class ErroIntegridade(Exception):
    """Arquivo de entrada alterado/ausente em relação ao manifesto."""


class ErroProvisorio(Exception):
    """Manifesto marcado como provisório (instância não oficial)."""


def sha256_arquivo(caminho) -> str:
    h = hashlib.sha256()
    with open(caminho, "rb") as f:
        for bloco in iter(lambda: f.read(65536), b""):
            h.update(bloco)
    return h.hexdigest()


def sha256_conjunto(hashes: dict) -> str:
    """Mesma regra do notebook: SHA-256 da concatenação dos hashes ordenados pelo nome do arquivo."""
    return hashlib.sha256("".join(hashes[k] for k in sorted(hashes)).encode("utf-8")).hexdigest()


def verificar_integridade(pasta, manifesto: dict) -> dict:
    pasta = Path(pasta)
    rel = {"arquivos": {}, "conjunto_ok": True, "id_ok": True}
    esperados = manifesto.get("arquivos", {})
    if not esperados:
        raise ErroIntegridade("manifesto_dados.json não possui o campo 'arquivos'.")
    for nome, esperado in esperados.items():
        caminho = pasta / nome
        calc = sha256_arquivo(caminho) if caminho.exists() else None
        rel["arquivos"][nome] = {"esperado": esperado, "calculado": calc, "ok": calc == esperado}
    if "sha256_conjunto" in manifesto:
        rel["conjunto_ok"] = sha256_conjunto(esperados) == manifesto["sha256_conjunto"]
    if "id_instancia" in manifesto and "sha256_conjunto" in manifesto:
        rel["id_ok"] = manifesto["sha256_conjunto"][:12] == manifesto["id_instancia"]
    rel["ok"] = all(v["ok"] for v in rel["arquivos"].values()) and rel["conjunto_ok"] and rel["id_ok"]
    if not rel["ok"]:
        ruins = [n for n, v in rel["arquivos"].items() if not v["ok"]]
        raise ErroIntegridade(f"Integridade violada. Arquivos divergentes/ausentes: {ruins or 'hash do conjunto/ID'}. "
                              "Regenere com o notebook oficial; não edite os arquivos.")
    return rel


def _exigir_colunas(df: pd.DataFrame, cols: list, nome: str) -> pd.DataFrame:
    if "idade_estoque" in df.columns and "idade_estoque_dias" not in df.columns:
        df = df.rename(columns={"idade_estoque": "idade_estoque_dias"})
    faltam = [c for c in cols if c not in df.columns]
    if faltam:
        raise ValueError(f"{nome}: colunas ausentes: {faltam}")
    return df


def validar_catalogo_q1(df: pd.DataFrame) -> pd.DataFrame:
    df = _exigir_colunas(df, COLUNAS_Q1, "catálogo Q1")
    _validar_comum(df, "estoque", "demanda_base_90d", "custo_logistico", "Q1")
    return df


def validar_catalogo_q2(df: pd.DataFrame) -> pd.DataFrame:
    df = _exigir_colunas(df, COLUNAS_Q2, "catálogo Q2")
    _validar_comum(df, "estoque_inicial", "demanda_base_semanal", "custo_logistico_base", "Q2")
    if (df["preco_concorrente_base"] <= 0).any():
        raise ValueError("Q2: preco_concorrente_base deve ser positivo.")
    return df


def _validar_comum(df, col_estoque, col_demanda, col_frete, q):
    if df.empty:
        raise ValueError(f"{q}: catálogo vazio.")
    if df["sku"].duplicated().any():
        raise ValueError(f"{q}: SKU duplicado.")
    for c in ["custo_unitario", "preco_atual", col_estoque, col_demanda, "elasticidade"]:
        if (df[c] <= 0).any():
            raise ValueError(f"{q}: coluna '{c}' deve ser estritamente positiva.")
    if (df[col_frete] < 0).any():
        raise ValueError(f"{q}: {col_frete} não pode ser negativo.")
    if not df["taxa_marketplace"].between(0, 1, inclusive="left").all():
        raise ValueError(f"{q}: taxa_marketplace deve estar em [0, 1).")
    if df.isna().any().any():
        raise ValueError(f"{q}: valores ausentes no catálogo.")


def validar_config_q1(cfg: dict) -> dict:
    for k in ["meta_caixa_90d", "divida_total_90d", "orcamento_promocional", "margem_minima_percentual",
              "niveis_preco_relativos"]:
        if k not in cfg:
            raise ValueError(f"config Q1: chave ausente '{k}'")
    if not cfg["niveis_preco_relativos"] or min(cfg["niveis_preco_relativos"]) <= 0:
        raise ValueError("config Q1: níveis de preço devem ser positivos.")
    if not 0 <= cfg["margem_minima_percentual"] < 1:
        raise ValueError("config Q1: margem mínima deve estar em [0, 1).")
    if cfg["orcamento_promocional"] < 0 or cfg["meta_caixa_90d"] <= 0:
        raise ValueError("config Q1: meta deve ser positiva e orçamento não negativo.")
    return cfg


def validar_config_q2(cfg: dict) -> dict:
    for k in ["numero_semanas", "caixa_inicial", "dividas", "variabilidade", "orcamento_promocional_semanal",
              "niveis_preco_relativos"]:
        if k not in cfg:
            raise ValueError(f"config Q2: chave ausente '{k}'")
    if cfg["numero_semanas"] < 1:
        raise ValueError("config Q2: numero_semanas deve ser >= 1")
    for s in ["sigma_demanda", "sigma_preco_concorrente", "sigma_custo_logistico"]:
        if cfg["variabilidade"].get(s, -1) < 0:
            raise ValueError(f"config Q2: variabilidade.{s} ausente ou negativa")
    return cfg


@dataclass
class DadosOficiais:
    pasta: Path
    manifesto: dict
    q1_catalogo: pd.DataFrame
    q1_config: dict
    q2_catalogo: pd.DataFrame
    q2_config: dict
    integridade: dict

    @property
    def grupo(self): return self.manifesto.get("grupo", self.q1_config.get("grupo"))
    @property
    def seed_base(self): return int(self.manifesto.get("seed_base", self.q1_config.get("seed_base", 0)))
    @property
    def seed_final(self): return int(self.manifesto.get("seed_final", self.q1_config.get("seed_final", 0)))
    @property
    def id_instancia(self): return self.manifesto.get("id_instancia") or self.manifesto.get("sha256_conjunto", "")[:12]
    @property
    def provisorio(self): return bool(self.manifesto.get("provisorio", False))


def _achar(arquivos: dict, prefixo: str, ext: str, pasta: Path) -> Path:
    nomes = [n for n in arquivos if n.startswith(prefixo) and n.endswith(ext)]
    if len(nomes) != 1:
        raise FileNotFoundError(f"Esperava exatamente 1 arquivo '{prefixo}*{ext}' no manifesto; achei {nomes}")
    return pasta / nomes[0]


def carregar_dados(pasta=None, verificar: bool = True, permitir_provisorio: bool = False) -> DadosOficiais:
    pasta = Path(pasta) if pasta else PASTA_DADOS
    mpath = pasta / "manifesto_dados.json"
    if not mpath.exists():
        raise FileNotFoundError(f"{mpath} não encontrado. Gere os arquivos com o notebook oficial e copie-os para {pasta}/")
    manifesto = json.loads(mpath.read_text(encoding="utf-8"))
    if manifesto.get("provisorio") and not permitir_provisorio:
        raise ErroProvisorio("Os dados em data/ são PROVISÓRIOS (manifesto marcado). Substitua pelos arquivos oficiais "
                             "gerados pelo notebook do professor ou use --permitir-provisorio apenas para testes.")
    integr = verificar_integridade(pasta, manifesto) if verificar else {"ok": None, "ignorada": True}
    arq = manifesto.get("arquivos", {})
    q1c = validar_catalogo_q1(pd.read_csv(_achar(arq, "q1_catalogo", ".csv", pasta)))
    q2c = validar_catalogo_q2(pd.read_csv(_achar(arq, "q2_catalogo", ".csv", pasta)))
    q1g = validar_config_q1(json.loads(_achar(arq, "q1_config", ".json", pasta).read_text(encoding="utf-8")))
    q2g = validar_config_q2(json.loads(_achar(arq, "q2_config", ".json", pasta).read_text(encoding="utf-8")))
    return DadosOficiais(pasta, manifesto, q1c, q1g, q2c, q2g, integr)
