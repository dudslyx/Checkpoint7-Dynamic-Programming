"""Fábricas de instâncias minúsculas e determinísticas para os testes (independentes dos dados oficiais)."""
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parents[1]
if str(RAIZ) not in sys.path:
    sys.path.insert(0, str(RAIZ))

from comum.dados import sha256_arquivo, sha256_conjunto  # noqa: E402


def catalogo_q1(n=6):
    rng = np.random.default_rng(3)
    return pd.DataFrame({
        "sku": [f"Q1-{i:03d}" for i in range(1, n + 1)], "categoria": ["Moda"] * n,
        "custo_unitario": rng.uniform(20, 200, n).round(2), "preco_atual": (rng.uniform(260, 500, n)).round(2),
        "estoque": rng.integers(20, 120, n), "demanda_base_90d": rng.integers(20, 150, n),
        "elasticidade": rng.uniform(0.8, 2.2, n).round(2), "taxa_marketplace": rng.uniform(0.08, 0.18, n).round(4),
        "custo_logistico": rng.uniform(5, 30, n).round(2), "idade_estoque_dias": rng.integers(5, 360, n)})


def config_q1(**kw):
    c = {"grupo": "G99", "seed_base": 1, "seed_final": 99001, "horizonte_dias": 90, "meta_caixa_90d": 1000.0, "divida_total_90d": 900.0,
         "orcamento_promocional": 800.0, "margem_minima_percentual": 0.05, "niveis_preco_relativos": [0.8, 0.9, 1.0, 1.1]}
    c.update(kw)
    return c


def catalogo_q2(n=4):
    rng = np.random.default_rng(5)
    return pd.DataFrame({
        "sku": [f"Q2-{i:03d}" for i in range(1, n + 1)], "categoria": ["Moda"] * n, "custo_unitario": rng.uniform(20, 80, n).round(2),
        "preco_atual": rng.uniform(120, 200, n).round(2), "estoque_inicial": rng.integers(20, 60, n),
        "demanda_base_semanal": rng.integers(3, 8, n), "elasticidade": rng.uniform(1.0, 2.0, n).round(2),
        "taxa_marketplace": rng.uniform(0.08, 0.18, n).round(4), "custo_logistico_base": rng.uniform(5, 15, n).round(2),
        "preco_concorrente_base": rng.uniform(120, 200, n).round(2)})


def config_q2(**kw):
    c = {"grupo": "G99", "seed_base": 1, "seed_final": 99001, "numero_semanas": 12, "caixa_inicial": 500.0,
         "dividas": {"semana_4": 800.0, "semana_8": 1000.0, "semana_12": 1500.0},
         "variabilidade": {"sigma_demanda": 0.1, "sigma_preco_concorrente": 0.05, "sigma_custo_logistico": 0.08},
         "orcamento_promocional_semanal": 40.0, "niveis_preco_relativos": [0.9, 1.0, 1.1], "n_simulacoes_recomendadas": 2000, "limite_risco_falha": 0.1}
    c.update(kw)
    return c


def escrever_instancia(pasta, provisorio=False, q1=None, q2=None):
    pasta = Path(pasta)
    arqs = {"q1_catalogo_G99.csv": (q1 if q1 is not None else catalogo_q1()), "q2_catalogo_G99.csv": (q2 if q2 is not None else catalogo_q2())}
    for nome, df in arqs.items():
        df.to_csv(pasta / nome, index=False)
    (pasta / "q1_config_G99.json").write_text(json.dumps(config_q1()), encoding="utf-8")
    (pasta / "q2_config_G99.json").write_text(json.dumps(config_q2()), encoding="utf-8")
    h = {n: sha256_arquivo(pasta / n) for n in ["q1_catalogo_G99.csv", "q1_config_G99.json", "q2_catalogo_G99.csv", "q2_config_G99.json"]}
    hc = sha256_conjunto(h)
    man = {"grupo": "G99", "seed_base": 1, "seed_final": 99001, "id_instancia": hc[:12], "arquivos": h, "sha256_conjunto": hc}
    if provisorio:
        man["provisorio"] = True
    (pasta / "manifesto_dados.json").write_text(json.dumps(man), encoding="utf-8")
    return man


def pasta_temporaria():
    return tempfile.TemporaryDirectory()
