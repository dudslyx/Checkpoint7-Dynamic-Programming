"""Ajustes de 'cenário-surpresa' aplicados por parâmetro (sem alterar código)."""
from __future__ import annotations

import argparse
from dataclasses import dataclass


@dataclass(frozen=True)
class Ajustes:
    meta_fator: float = 1.0            # Q1: meta de caixa x fator (ex.: 1.10 = +10%)
    divida_fator: float = 1.0          # Q2: dívidas x fator
    custo_logistico_fator: float = 1.0  # Q1 e Q2: custo logístico x fator (ex.: 1.15)
    orcamento_fator: float = 1.0       # Q1 e Q2: orçamento promocional x fator (ex.: 0.75)
    concorrente_delta: float = 0.0     # Q2: variação do preço do concorrente (ex.: -0.08)
    skus_indisponiveis: tuple = ()     # remove SKUs do catálogo

    def descricao(self) -> list[str]:
        d = []
        if self.meta_fator != 1.0: d.append(f"meta de caixa x{self.meta_fator:g}")
        if self.divida_fator != 1.0: d.append(f"dívidas x{self.divida_fator:g}")
        if self.custo_logistico_fator != 1.0: d.append(f"custo logístico x{self.custo_logistico_fator:g}")
        if self.orcamento_fator != 1.0: d.append(f"orçamento promocional x{self.orcamento_fator:g}")
        if self.concorrente_delta != 0.0: d.append(f"preço do concorrente {self.concorrente_delta:+.0%}")
        if self.skus_indisponiveis: d.append("SKUs indisponíveis: " + ", ".join(self.skus_indisponiveis))
        return d


def adicionar_argumentos(p: argparse.ArgumentParser) -> None:
    p.add_argument("--dados", default=None, help="pasta com os arquivos oficiais (padrão: data/)")
    p.add_argument("--saida", default=None, help="pasta de resultados (padrão: <questao>/resultados)")
    p.add_argument("--permitir-provisorio", action="store_true", help="aceita manifesto marcado como provisório")
    p.add_argument("--ignorar-integridade", action="store_true", help="NÃO recomendado: pula a checagem SHA-256")
    g = p.add_argument_group("cenário-surpresa (tudo por parâmetro)")
    g.add_argument("--meta-fator", type=float, default=1.0)
    g.add_argument("--divida-fator", type=float, default=1.0)
    g.add_argument("--custo-logistico-fator", type=float, default=1.0)
    g.add_argument("--orcamento-fator", type=float, default=1.0)
    g.add_argument("--concorrente-delta", type=float, default=0.0)
    g.add_argument("--sku-indisponivel", action="append", default=[], metavar="SKU")


def ajustes_de_args(a) -> Ajustes:
    return Ajustes(a.meta_fator, a.divida_fator, a.custo_logistico_fator, a.orcamento_fator,
                   a.concorrente_delta, tuple(a.sku_indisponivel))
