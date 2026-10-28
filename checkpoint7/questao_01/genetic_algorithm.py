"""Etapa B — Algoritmo Genético para a política de preços do catálogo completo.

Cromossomo: vetor de inteiros; gene i = posição do preço escolhido na lista de preços PERMITIDOS do SKU i
(o piso de margem é respeitado por construção — sem penalidades, sem soluções inviáveis).
Fitness: caixa total (soma de tabela[i, gene_i]); avaliação O(n) por indivíduo, vetorizada na população.
Operadores: seleção por torneio, cruzamento uniforme, mutação (passo ±1 ou sorteio), elitismo, parada por paciência.
Complexidade: tempo O(G*P*n); memória O(P*n + n*K).
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np


@dataclass(frozen=True)
class ConfigGA:
    nome: str = "A (principal)"
    populacao: int = 120
    geracoes: int = 250
    torneio_k: int = 3
    p_cruzamento: float = 0.9
    p_mutacao: float = 0.05        # probabilidade de mutação POR GENE
    elitismo: int = 4
    paciencia: int | None = 40     # gerações sem melhoria para parar (None = roda todas)


CONFIG_A = ConfigGA()
CONFIG_B = ConfigGA(nome="B (enxuta)", populacao=40, torneio_k=2, p_cruzamento=0.8, p_mutacao=0.01, elitismo=1)


@dataclass
class ResultadoGA:
    melhor_caixa: float
    melhores_genes: np.ndarray
    hist_melhor: np.ndarray
    hist_media: np.ndarray
    geracoes_executadas: int
    avaliacoes: int
    geracao_melhor: int          # 1ª geração em que o melhor valor final foi atingido
    config: ConfigGA = field(default_factory=ConfigGA)


def avaliar_populacao(tabela: np.ndarray, pop: np.ndarray) -> np.ndarray:
    """Fitness vetorizado: um 'gather' na tabela pré-computada. O(P*n)."""
    return tabela[np.arange(pop.shape[1]), pop].sum(axis=1)


def avaliar_populacao_ingenua(tabela: list, pop: list) -> list:
    """Versão 'ANTES': laços Python puros (usada só na comparação de otimização)."""
    return [sum(tabela[i][g] for i, g in enumerate(ind)) for ind in pop]


def executar_ga(tabela: np.ndarray, k: np.ndarray, cfg: ConfigGA, rng: np.random.Generator) -> ResultadoGA:
    k = np.asarray(k, dtype=np.int64)
    n = len(k)
    if n == 0:
        return ResultadoGA(0.0, np.zeros(0, dtype=np.int64), np.zeros(1), np.zeros(1), 0, 0, 0, cfg)
    P = cfg.populacao + (cfg.populacao % 2)                      # população par (pares de pais)
    E = min(cfg.elitismo, P)
    pop = (rng.random((P, n)) * k).astype(np.int64)
    melhor_val, melhor_ind = -np.inf, pop[0].copy()
    hb, hm, sem_melhoria, ger_melhor = [], [], 0, 0
    for g in range(cfg.geracoes):
        fit = avaliar_populacao(tabela, pop)
        j = int(np.argmax(fit))
        if fit[j] > melhor_val + 1e-9:
            melhor_val, melhor_ind, sem_melhoria, ger_melhor = float(fit[j]), pop[j].copy(), 0, g + 1
        else:
            sem_melhoria += 1
        hb.append(melhor_val)
        hm.append(float(fit.mean()))
        if cfg.paciencia is not None and sem_melhoria >= cfg.paciencia:
            break
        elite = pop[np.argsort(-fit)[:E]].copy()
        cand = rng.integers(0, P, size=(P, cfg.torneio_k))                 # seleção por torneio
        vencedor = cand[np.arange(P), np.argmax(fit[cand], axis=1)]
        pais = pop[vencedor]
        a, b = pais[0::2], pais[1::2]                                       # cruzamento uniforme
        troca = (rng.random(a.shape) < 0.5) & (rng.random((a.shape[0], 1)) < cfg.p_cruzamento)
        filhos = np.empty_like(pais)
        filhos[0::2], filhos[1::2] = np.where(troca, b, a), np.where(troca, a, b)
        mut = rng.random(filhos.shape) < cfg.p_mutacao                       # mutação: passo ±1 (50%) ou sorteio
        passo = np.minimum(np.maximum(filhos + (rng.integers(0, 2, filhos.shape) * 2 - 1), 0), k - 1)
        sorteio = (rng.random(filhos.shape) * k).astype(np.int64)
        novo = np.where(rng.random(filhos.shape) < 0.5, passo, sorteio)
        filhos = np.where(mut, novo, filhos)
        filhos[:E] = elite                                                   # elitismo
        pop = filhos
    ger = len(hb)
    return ResultadoGA(melhor_val, melhor_ind, np.array(hb), np.array(hm), ger, ger * P, ger_melhor, cfg)


def com_parametros(cfg: ConfigGA, populacao=None, mutacao=None) -> ConfigGA:
    """Cenário-surpresa: troca população/mutação por parâmetro."""
    return replace(cfg, populacao=populacao or cfg.populacao, p_mutacao=cfg.p_mutacao if mutacao is None else mutacao)
