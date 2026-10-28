"""Geradores pseudoaleatórios reprodutíveis: um fluxo independente por componente."""
import zlib

import numpy as np


def criar_rng(seed: int, fluxo: str) -> np.random.Generator:
    """RNG determinístico a partir de (seed, nome do fluxo).

    Fluxos com nomes diferentes (ex.: 'ga', 'mc-estimativa', 'mc-validacao-3') são
    estatisticamente independentes, mesmo usando a mesma seed da instância.
    """
    return np.random.default_rng(np.random.SeedSequence([int(seed), zlib.crc32(fluxo.encode("utf-8"))]))
