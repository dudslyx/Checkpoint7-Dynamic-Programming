"""Questão 2 — Plano de Sobrevivência em 12 Semanas (Monte Carlo -> DP -> Greedy -> Monte Carlo de validação).

Exemplos:
    python questao_02/main.py --seed 1020 --cenarios 5000
    python questao_02/main.py --modo rapido --concorrente-delta -0.08 --custo-logistico-fator 1.15
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from comum import graficos
from comum.cenarios import adicionar_argumentos, ajustes_de_args
from comum.dados import carregar_dados
from comum.metricas import medir
from questao_02 import benchmark
from questao_02.dynamic_programming import reconstruir_caminho, resolver_dp
from questao_02.greedy import alocar_promocoes
from questao_02.modelo import construir_instancia
from questao_02.monte_carlo import (estimar_estatisticas, pagamentos, politica_adaptativa, politica_caminho, politica_fixa,
                                    resumir, simular)

PASTA = Path(__file__).resolve().parent
THETAS = (0.0, 0.5, 1.0, 2.0)


def _py(o):
    if isinstance(o, dict): return {str(k): _py(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)): return [_py(v) for v in o]
    if isinstance(o, np.ndarray): return _py(o.tolist())
    if isinstance(o, np.integer): return int(o)
    if isinstance(o, np.floating): return float(o)
    if isinstance(o, np.bool_): return bool(o)
    return o


def _linha(nome, r):
    d = {"politica": nome}
    for s, v in r["p_pagamento"].items(): d[f"P_pagar_semana_{s}"] = v
    d.update({"P_pagar_todas": r["p_todas"], "cenarios_falha": r["cenarios_falha"], "caixa_acumulado_medio_rs": r["caixa_acumulado_medio"],
              "saldo_liquido_medio_rs": r["saldo_liquido_medio"], f"saldo_liquido_P{int(r['percentil_risco'])}_rs": r["saldo_liquido_percentil"],
              "cvar_rs": r["cvar"], "margem_pct": r["margem_pct"], "estoque_final_un": r["estoque_final_un"],
              "estoque_final_pct": r["estoque_final_pct"], "risco_ok": r["risco_ok"]})
    return d


def ruptura(inst, caixa_sem, limite):
    """Maior multiplicador das dívidas com P(pagar todas) >= 1 - limite (bisseção; o fluxo de caixa não depende da dívida)."""
    lo, hi = 1.0, 8.0
    def ok(m):
        _, pg = pagamentos(replace(inst, dividas=inst.dividas * m), caixa_sem)
        return pg.all(1).mean() >= 1 - limite
    if not ok(lo): return 1.0
    if ok(hi): return hi
    for _ in range(30):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if ok(mid) else (lo, mid)
    return lo


def executar(dados, ajustes, seed, n_val=5000, n_constr=None, modo="completo", saida=None) -> dict:
    saida = Path(saida) if saida else PASTA / "resultados"
    saida.mkdir(parents=True, exist_ok=True)
    inst = construir_instancia(dados.q2_catalogo, dados.q2_config, ajustes)
    n_constr = max(2000, n_constr or inst.n_sim)
    n_val = max(5000, n_val)
    n, W = inst.n, inst.W

    # A) Monte Carlo — estimativa (>= 2000 cenários) ------------------------------------------------------------
    m_est = medir(estimar_estatisticas, inst, n_constr, seed)
    est = m_est.resultado

    # B + C + A) calibração da urgência (theta): DP -> Greedy -> Monte Carlo (cenários de CONSTRUÇÃO) --------------
    cal = []
    for th in THETAS:
        s_ = resolver_dp(inst, est, th)
        c_ = reconstruir_caminho(inst, est, s_)
        p_ = alocar_promocoes(inst, est, c_, s_)
        r_ = resumir(inst, simular(inst, politica_adaptativa(s_.politica), p_, n_constr, seed, "calibracao"))
        cal.append({"theta": th, "P_pagar_todas": r_["p_todas"], "caixa_acumulado_medio_rs": r_["caixa_acumulado_medio"],
                    "saldo_liquido_P10_rs": r_["saldo_liquido_percentil"], "pesos_semana": ",".join(f"{v:.2f}" for v in s_.pesos[[0, 4, 8]])})
    dfc = pd.DataFrame(cal)
    theta = float(dfc.sort_values(["P_pagar_todas", "caixa_acumulado_medio_rs", "theta"], ascending=[False, False, True]).iloc[0]["theta"])
    dfc["escolhido"] = dfc["theta"] == theta
    dfc.to_csv(saida / "calibracao_theta.csv", index=False)

    # B) DP final, trajetória ótima e C) Greedy ------------------------------------------------------------------
    m_dp = medir(resolver_dp, inst, est, theta)
    sol = m_dp.resultado
    cam = reconstruir_caminho(inst, est, sol)
    m_gr = medir(alocar_promocoes, inst, est, cam, sol)
    plano = m_gr.resultado

    # A') validação: 5000 cenários independentes (fluxo 'validacao' != 'mc-estimativa'/'calibracao') ----------------
    rep = [int(i) for i in _representativos(inst, cam)]
    pol_ad = politica_adaptativa(sol.politica)
    m_val = medir(simular, inst, pol_ad, plano, n_val, seed, "validacao", False, rep)
    res_plano = m_val.resultado
    res_base = simular(inst, politica_fixa(inst.ref), None, n_val, seed, "validacao")
    res_cam = simular(inst, politica_caminho(cam.niveis), None, n_val, seed, "validacao")
    res_ad = simular(inst, pol_ad, None, n_val, seed, "validacao")
    res_pers = simular(inst, pol_ad, plano, n_val, seed, "validacao", True)
    resumos = {"Base: preço atual fixo (nível 1,00)": resumir(inst, res_base),
               "DP — trajetória fixa (malha aberta), sem promoção": resumir(inst, res_cam),
               "DP — reprecificação semanal adaptativa, sem promoção": resumir(inst, res_ad),
               "PLANO COMPLETO: DP adaptativa + Greedy": resumir(inst, res_plano),
               "Plano completo sob choque PERSISTENTE do concorrente": resumir(inst, res_pers)}
    pd.DataFrame([_linha(k, v) for k, v in resumos.items()]).to_csv(saida / "comparativo_politicas.csv", index=False)
    rp = resumos["PLANO COMPLETO: DP adaptativa + Greedy"]
    z = 1.96
    linhas_p = []
    for j, s in enumerate(inst.semanas_pag):
        p_ = rp["p_pagamento"][s]
        linhas_p.append({"compromisso": f"semana {s}", "valor_rs": inst.dividas[j], "P_pagar_no_prazo": p_,
                         "IC95_inf": max(0, p_ - z * np.sqrt(p_ * (1 - p_) / n_val)), "IC95_sup": min(1, p_ + z * np.sqrt(p_ * (1 - p_) / n_val)),
                         "P_falha": 1 - p_, "limite_falha": inst.limite_risco, "dentro_do_limite": (1 - p_) <= inst.limite_risco})
    linhas_p.append({"compromisso": "as três juntas", "valor_rs": inst.dividas.sum(), "P_pagar_no_prazo": rp["p_todas"], "IC95_inf": np.nan,
                     "IC95_sup": np.nan, "P_falha": 1 - rp["p_todas"], "limite_falha": inst.limite_risco, "dentro_do_limite": (1 - rp["p_todas"]) <= inst.limite_risco})
    pd.DataFrame(linhas_p).to_csv(saida / "prob_pagamento.csv", index=False)

    # distribuição do caixa final -----------------------------------------------------------------------------------
    lb, lp = resumos["Base: preço atual fixo (nível 1,00)"]["liquido"], rp["liquido"]
    qs = [0, 1, 5, 10, 25, 50, 75, 90, 95, 99, 100]
    pd.DataFrame({"percentil": qs, "base_fixa_rs": np.percentile(lb, qs), "plano_completo_rs": np.percentile(lp, qs)}).to_csv(
        saida / "distribuicao_caixa_final.csv", index=False)
    graficos.histograma(saida / "distribuicao_caixa_final.png", {"Base (preço fixo)": lb, "Plano completo": lp},
                        "Saldo final = caixa inicial + caixa 12 semanas − dívidas (5.000 cenários de validação)", "R$",
                        {"P10 plano": rp["saldo_liquido_percentil"], "saldo 0": 0.0})

    # trajetórias de preço/estoque de SKUs representativos ----------------------------------------------------------
    itens, linhas_t = [], []
    for c, i in enumerate(rep):
        niv_med = inst.niveis[res_plano.traj_niv[:, :, c]].mean(0)
        est_m, p10, p90 = res_plano.traj_est[:, :, c].mean(0), np.percentile(res_plano.traj_est[:, :, c], 10, axis=0), np.percentile(res_plano.traj_est[:, :, c], 90, axis=0)
        itens.append((f"{inst.sku[i]} ({inst.categoria[i]})", niv_med, est_m, p10, p90))
        for w in range(W):
            linhas_t.append({"sku": inst.sku[i], "categoria": inst.categoria[i], "semana": w + 1, "nivel_caminho_otimo": inst.niveis[cam.niveis[w, i]],
                             "preco_caminho_otimo": inst.preco[i] * inst.niveis[cam.niveis[w, i]], "estoque_esperado_inicio": cam.estoque_ini[w, i],
                             "vendas_esperadas": cam.vendas_esp[w, i], "promocao_rs": plano.gasto[w, i], "nivel_medio_simulado": niv_med[w],
                             "estoque_medio_sim_fim": est_m[w + 1], "estoque_P10_fim": p10[w + 1], "estoque_P90_fim": p90[w + 1]})
    pd.DataFrame(linhas_t).to_csv(saida / "trajetoria_preco_estoque.csv", index=False)
    graficos.trajetorias(saida / "trajetoria_preco_estoque.png", np.arange(1, W + 1), itens)

    # plano completo (todos os SKUs x semanas) ------------------------------------------------------------------------
    linhas_pl = [{"sku": inst.sku[i], "semana": w + 1, "nivel_preco": inst.niveis[cam.niveis[w, i]], "preco": inst.preco[i] * inst.niveis[cam.niveis[w, i]],
                  "estoque_esperado_inicio": cam.estoque_ini[w, i], "vendas_esperadas": cam.vendas_esp[w, i], "caixa_esperado": cam.caixa_esp[w, i],
                  "promocao_rs": plano.gasto[w, i], "lift_demanda": plano.lift[w, i]} for w in range(W) for i in range(n)]
    pd.DataFrame(linhas_pl).to_csv(saida / "plano_semanal.csv", index=False)

    # estresse (cenários-surpresa SEM replanejar) + ponto de ruptura --------------------------------------------------
    estresse = []
    def avalia(nome, ins_):
        a = resumir(ins_, simular(ins_, politica_fixa(ins_.ref), None, n_val, seed, "validacao"))
        b = resumir(ins_, simular(ins_, pol_ad, plano, n_val, seed, "validacao"))
        estresse.append({"cenario": nome, "P_todas_base": a["p_todas"], "P_todas_plano": b["p_todas"], "saldo_medio_base_rs": a["saldo_liquido_medio"],
                         "saldo_medio_plano_rs": b["saldo_liquido_medio"], "plano_dentro_do_limite": b["risco_ok"]})
    avalia("Cenário atual", inst)
    avalia("Concorrente −8%", replace(inst, fator_conc=inst.fator_conc * 0.92))
    avalia("Concorrente −15%", replace(inst, fator_conc=inst.fator_conc * 0.85))
    avalia("Custo logístico +15%", replace(inst, frete0=inst.frete0 * 1.15))
    avalia("Custo logístico +30%", replace(inst, frete0=inst.frete0 * 1.30))
    for f in (1.2, 1.4, 1.6, 1.8, 2.0):
        avalia(f"Dívidas +{int(round((f - 1) * 100))}%", replace(inst, dividas=inst.dividas * f))
    avalia("Combinado: concorrente −8%, logística +15%, dívidas +20%", replace(inst, fator_conc=inst.fator_conc * 0.92, frete0=inst.frete0 * 1.15, dividas=inst.dividas * 1.2))
    pd.DataFrame(estresse).to_csv(saida / "estresse.csv", index=False)
    rup = {"base": ruptura(inst, res_base.caixa_sem, inst.limite_risco), "plano": ruptura(inst, res_plano.caixa_sem, inst.limite_risco)}

    final = {
        "instancia": {"grupo": dados.grupo, "seed_base": dados.seed_base, "seed_final": dados.seed_final, "seed_execucao": seed,
                      "id_instancia": dados.id_instancia, "provisoria": dados.provisorio, "n_skus": n, "semanas": W, "ajustes": ajustes.descricao(),
                      "dividas": {str(s): float(v) for s, v in zip(inst.semanas_pag, inst.dividas)}, "caixa_inicial": inst.caixa_inicial,
                      "orcamento_promocional_semanal": inst.orcamento_sem, "sigma": {"demanda": inst.sig_d, "concorrente": inst.sig_c, "logistica": inst.sig_f},
                      "limite_risco_falha": inst.limite_risco, "cenarios_construcao": n_constr, "cenarios_validacao": n_val},
        "theta_escolhido": theta, "pesos_semana": sol.pesos,
        "dp": {"estados": sol.estados, "leituras_V": sol.leituras, "reuso_por_estado": sol.leituras / sol.estados, "caminhos_ingenuos_L_pow_W": sol.caminhos_ingenuos},
        "greedy": {"gasto_total": float(plano.gasto.sum()), "orcamento_total": inst.orcamento_sem * W, "uso_orcamento_pct": float(100 * plano.gasto.sum() / (inst.orcamento_sem * W)),
                   "ganho_esperado_dp": float(plano.ganho_esperado.sum())},
        "plano_completo": {k: v for k, v in rp.items() if k != "liquido"},
        "base": {k: v for k, v in resumos["Base: preço atual fixo (nível 1,00)"].items() if k != "liquido"},
        "ruptura_multiplicador_dividas": rup,
        "desempenho": {"estimativa": {"tempo_s": m_est.tempo_s, "pico_kib": m_est.pico_kib}, "dp": {"tempo_s": m_dp.tempo_s, "pico_kib": m_dp.pico_kib},
                       "greedy": {"tempo_s": m_gr.tempo_s, "pico_kib": m_gr.pico_kib}, "validacao": {"tempo_s": m_val.tempo_s, "pico_kib": m_val.pico_kib}},
        "representativos": [inst.sku[i] for i in rep],
    }
    if modo == "completo":
        fi, fc = benchmark.escala_instancia(inst, seed), benchmark.escala_cenarios(inst, seed)
        fi.to_csv(saida / "bench_instancia.csv", index=False)
        fc.to_csv(saida / "bench_cenarios.csv", index=False)
        piv = fi.pivot(index="n_skus", columns="etapa", values="tempo_s")
        graficos.escala(saida / "bench_instancia.png", piv.index, {c: piv[c] for c in piv.columns}, "n (SKUs)", "tempo (s)", "Q2: tempo por etapa x nº de SKUs", logx=True, logy=True)
    (saida / "resultado_final.json").write_text(json.dumps(_py(final), ensure_ascii=False, indent=2), encoding="utf-8")
    return final


def _representativos(inst, cam, k=4):
    ordem = np.argsort(-cam.caixa_esp.sum(0))
    cats, esc = set(), []
    for i in ordem:
        if inst.categoria[i] not in cats:
            cats.add(inst.categoria[i]); esc.append(i)
        if len(esc) == k: break
    for i in ordem:
        if len(esc) == k: break
        if i not in esc: esc.append(i)
    return esc[:k]


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    adicionar_argumentos(p)
    p.add_argument("--seed", type=int, default=None, help="seed (padrão: seed_final da instância)")
    p.add_argument("--cenarios", type=int, default=5000, help="cenários da validação final (>= 5000)")
    p.add_argument("--cenarios-construcao", type=int, default=None, help="cenários da estimativa (>= 2000)")
    p.add_argument("--modo", choices=["rapido", "completo"], default="completo")
    a = p.parse_args(argv)
    dados = carregar_dados(a.dados, verificar=not a.ignorar_integridade, permitir_provisorio=a.permitir_provisorio)
    seed = dados.seed_final if a.seed is None else a.seed
    r = executar(dados, ajustes_de_args(a), seed, a.cenarios, a.cenarios_construcao, a.modo, a.saida)
    i, pc = r["instancia"], r["plano_completo"]
    print(f"[Q2] grupo {i['grupo']} | seed {seed} | instância {i['id_instancia']}" + (" | PROVISÓRIA" if i["provisoria"] else ""))
    print("[Q2] P(pagar no prazo): " + " | ".join(f"sem {s}: {v:.1%}" for s, v in pc["p_pagamento"].items()) + f" | todas: {pc['p_todas']:.1%} | theta={r['theta_escolhido']}")
    print(f"[Q2] saldo líquido médio R$ {pc['saldo_liquido_medio']:,.2f} | P10 R$ {pc['saldo_liquido_percentil']:,.2f} | margem {pc['margem_pct']:.1f}% | estoque final {pc['estoque_final_pct']:.1f}%")
    return r


if __name__ == "__main__":
    main()
