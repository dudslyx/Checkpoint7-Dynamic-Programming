"""Gráficos (matplotlib, backend sem tela) usados nas evidências das duas questões."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch


def _salvar(fig, caminho):
    fig.tight_layout()
    fig.savefig(caminho, dpi=130)
    plt.close(fig)


def convergencia(caminho, series: dict, titulo: str):
    """series: nome -> (hist_melhor, hist_media). Fitness (caixa) por geração."""
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for i, (nome, (melhor, media)) in enumerate(series.items()):
        c = f"C{i}"
        ax.plot(np.arange(1, len(melhor) + 1), melhor, color=c, lw=2, label=f"{nome} — melhor")
        ax.plot(np.arange(1, len(media) + 1), media, color=c, lw=1, ls="--", alpha=.7, label=f"{nome} — média")
    ax.set_xlabel("Geração"); ax.set_ylabel("Fitness = caixa em 90 dias (R$)"); ax.set_title(titulo)
    ax.grid(alpha=.3); ax.legend(fontsize=8)
    _salvar(fig, caminho)


def escala(caminho, x, series: dict, xlabel, ylabel, titulo, logx=False, logy=False):
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for nome, y in series.items():
        ax.plot(x, y, marker="o", label=nome)
    if logx: ax.set_xscale("log")
    if logy: ax.set_yscale("log")
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel); ax.set_title(titulo); ax.grid(alpha=.3, which="both"); ax.legend(fontsize=8)
    _salvar(fig, caminho)


def histograma(caminho, dados: dict, titulo, xlabel, linhas: dict | None = None):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for i, (nome, v) in enumerate(dados.items()):
        ax.hist(v, bins=50, alpha=.55, color=f"C{i}", label=nome, density=True)
    for nome, v in (linhas or {}).items():
        ax.axvline(v, ls="--", color="k"); ax.text(v, ax.get_ylim()[1] * .92, " " + nome, rotation=90, va="top", fontsize=8)
    ax.set_xlabel(xlabel); ax.set_ylabel("densidade"); ax.set_title(titulo); ax.legend(fontsize=8); ax.grid(alpha=.3)
    _salvar(fig, caminho)


def trajetorias(caminho, semanas, itens: list):
    """itens: [(rotulo, niveis(W), estoque_medio(W+1), p10(W+1), p90(W+1))]; 2 linhas x n colunas."""
    n = len(itens)
    fig, axs = plt.subplots(2, n, figsize=(3.6 * n, 6), squeeze=False)
    for j, (rot, niv, est, p10, p90) in enumerate(itens):
        axs[0, j].step(semanas, niv, where="post", color="C0"); axs[0, j].set_title(rot, fontsize=9)
        axs[0, j].set_ylabel("nível de preço" if j == 0 else ""); axs[0, j].grid(alpha=.3)
        sem = np.arange(len(est))
        axs[1, j].plot(sem, est, color="C1", label="média"); axs[1, j].fill_between(sem, p10, p90, color="C1", alpha=.25, label="P10–P90")
        axs[1, j].set_xlabel("semana"); axs[1, j].set_ylabel("estoque (un.)" if j == 0 else ""); axs[1, j].grid(alpha=.3)
        if j == 0: axs[1, j].legend(fontsize=8)
    _salvar(fig, caminho)


def diagrama_fluxo(caminho):
    fig, ax = plt.subplots(figsize=(13, 6.2)); ax.axis("off"); ax.set_xlim(0, 13); ax.set_ylim(-0.3, 6.2)

    def caixa(x, y, txt, cor, w=2.3, h=0.9):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.05", fc=cor, ec="k"))
        ax.text(x + w / 2, y + h / 2, txt, ha="center", va="center", fontsize=8.5)
        return (x, y, w, h)

    def seta(a, b, txt="", rad=0.0):
        ax.annotate("", xy=(b[0], b[1]), xytext=(a[0], a[1]), arrowprops=dict(arrowstyle="->", lw=1.4, connectionstyle=f"arc3,rad={rad}"))
        if txt: ax.text((a[0] + b[0]) / 2, (a[1] + b[1]) / 2 + .12, txt, fontsize=7, ha="center")

    ax.text(.1, 5.95, "QUESTÃO 1 — Operação Caixa 90", fontsize=11, weight="bold")
    d1 = caixa(.1, 4.6, "q1_catalogo.csv\nq1_config.json\n(SHA-256 verificado)", "#eeeeee")
    bf = caixa(3.0, 4.95, "A. Brute Force\n(subconjunto m SKUs)\nótimo exato", "#ffd9b3")
    ga_s = caixa(3.0, 3.85, "B1. GA no mesmo\nsubconjunto", "#cfe8ff")
    ga = caixa(6.2, 4.4, "B2. Algoritmo Genético\ncatálogo completo\n→ política de preços", "#cfe8ff", w=2.6, h=1.1)
    gr = caixa(9.4, 4.4, "C. Greedy\nincentivos por benefício\nmarginal / real", "#d6f5d6", w=2.4, h=1.1)
    out = caixa(9.4, 3.2, "caixa, margem, estoque,\ncobertura da dívida,\nuso do orçamento", "#eeeeee", w=2.4, h=0.9)
    seta((2.4, 5.05), (3.0, 5.35)); seta((2.4, 4.95), (3.0, 4.3))
    seta((5.3, 5.3), (6.2, 4.95), "")
    seta((4.15, 3.85), (4.15, 3.85))
    ax.annotate("", xy=(4.15, 4.95), xytext=(4.15, 4.75), arrowprops=dict(arrowstyle="<->", lw=1.4))
    ax.text(4.25, 4.82, "gap %", fontsize=7)
    seta((8.8, 4.95), (9.4, 4.95), "política"); seta((10.6, 4.4), (10.6, 4.1))

    ax.text(.1, 2.55, "QUESTÃO 2 — Plano de Sobrevivência em 12 Semanas", fontsize=11, weight="bold")
    d2 = caixa(.1, 1.2, "q2_catalogo.csv\nq2_config.json\n(SHA-256 verificado)", "#eeeeee")
    mc1 = caixa(3.0, 1.2, "A. Monte Carlo\n(estimativa, ≥2000 cenários)\ndemanda × preço", "#ffd9b3", w=2.5)
    dp = caixa(6.1, 1.2, "B. Programação Dinâmica\nestado (semana, estoque)\n→ política de preços", "#cfe8ff", w=2.5)
    g2 = caixa(9.2, 1.2, "C. Greedy semanal\ndestaques / mídia\npor benefício marginal", "#d6f5d6", w=2.4)
    mc2 = caixa(9.2, 0.0, "A'. Monte Carlo (validação\n5000 cenários independentes)\nP(pagar semanas 4, 8, 12)", "#ffd9b3", w=3.4, h=0.85)
    seta((2.4, 1.65), (3.0, 1.65)); seta((5.5, 1.65), (6.1, 1.65), "estatísticas"); seta((8.6, 1.65), (9.2, 1.65), "plano")
    seta((10.4, 1.2), (10.4, .85), "plano completo")
    ax.text(.1, .45, "Cenários da validação são independentes dos usados na construção\n(fluxos de seed distintos). Urgência θ da DP calibrada só nos cenários de construção.", fontsize=8)
    _salvar(fig, caminho)
