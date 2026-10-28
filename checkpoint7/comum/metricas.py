"""Medição real de tempo (time.perf_counter) e memória de pico (tracemalloc)."""
from __future__ import annotations

import time
import tracemalloc
from dataclasses import dataclass
from typing import Any, Callable


@dataclass
class Medicao:
    tempo_s: float          # melhor tempo entre as repetições (sem tracemalloc ligado)
    pico_kib: float         # pico de memória alocada durante UMA execução (tracemalloc)
    resultado: Any = None


def medir(func: Callable, *args, repeticoes: int = 1, **kwargs) -> Medicao:
    """Executa `func` medindo tempo e pico de memória.

    Tempo e memória são medidos em execuções separadas porque o tracemalloc
    adiciona sobrecarga e distorceria o tempo.
    """
    melhor = float("inf")
    resultado = None
    for _ in range(max(1, repeticoes)):
        t0 = time.perf_counter()
        resultado = func(*args, **kwargs)
        melhor = min(melhor, time.perf_counter() - t0)
    tracemalloc.start()
    try:
        func(*args, **kwargs)
        _, pico = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    return Medicao(melhor, pico / 1024.0, resultado)
