import itertools
import unittest
from pathlib import Path

import numpy as np

from tests.helpers import catalogo_q2, config_q2, escrever_instancia, pasta_temporaria
from comum.cenarios import Ajustes
from comum.dados import carregar_dados
from questao_02.dynamic_programming import reconstruir_caminho, resolver_dp
from questao_02.greedy import alocar_promocoes
from questao_02.main import executar
from questao_02.modelo import InstanciaQ2, construir_instancia, peso_semana
from questao_02.monte_carlo import (EstatisticasMC, estimar_estatisticas, pagamentos, politica_adaptativa, politica_fixa,
                                    resumir, simular)


def inst_pequena(**aj):
    return construir_instancia(catalogo_q2(), config_q2(), Ajustes(**aj))


def inst_minima(theta_semanas=(3,)):
    """1 SKU, 3 semanas, 3 níveis: pequeno o bastante para enumerar todos os 3^3 caminhos."""
    z = np.zeros(1)
    return InstanciaQ2(sku=np.array(["A"]), categoria=np.array(["Moda"]), custo=z + 50, preco=z + 100, estoque0=np.array([6]), demanda0=z + 5,
                       elast=z + 1.5, taxa=z + 0.1, frete0=z + 5, preco_conc0=z + 100, niveis=np.array([0.9, 1.0, 1.1]), W=3,
                       semanas_pag=theta_semanas, dividas=np.array([100.0]), caixa_inicial=0.0, sig_d=0.0, sig_c=0.0, sig_f=0.0, orcamento_sem=0.0)


class TestMonteCarlo(unittest.TestCase):
    def test_reprodutivel_e_shapes(self):
        inst = inst_pequena()
        a, b = estimar_estatisticas(inst, 500, 11), estimar_estatisticas(inst, 500, 11)
        np.testing.assert_array_equal(a.dq, b.dq)
        self.assertEqual(a.dq.shape, (inst.n, inst.L, 40))
        c = estimar_estatisticas(inst, 500, 12)
        self.assertFalse(np.array_equal(a.dq, c.dq))

    def test_demanda_cai_com_preco(self):
        est = estimar_estatisticas(inst_pequena(), 1000, 1)
        self.assertTrue(np.all(np.diff(est.media_dem, axis=1) < 0))
        self.assertTrue(np.all(np.diff(est.dq, axis=2) >= 0))             # quantis crescentes

    def test_choques_tem_media_um(self):
        inst = inst_pequena()
        est = estimar_estatisticas(inst, 20000, 3)
        ref = inst.ref
        esperado = inst.demanda0 * inst.niveis[ref] ** (-inst.elast) * np.exp(0.5 * (inst.elast * inst.sig_c) ** 2)   # E[zeta^e]
        np.testing.assert_allclose(est.media_dem[:, ref], esperado, rtol=0.03)

    def test_simulacao_reprodutivel_independente_do_bloco(self):
        inst = inst_pequena()
        pol = politica_fixa(inst.ref)
        a = simular(inst, pol, None, 1200, 5, "t")
        b = simular(inst, pol, None, 1200, 5, "t")
        np.testing.assert_array_equal(a.caixa_sem, b.caixa_sem)
        self.assertTrue((a.est_final_un >= 0).all())

    def test_pagamentos_cascata(self):
        inst = inst_minima()
        inst = InstanciaQ2(**{**inst.__dict__, "W": 4, "semanas_pag": (2, 4), "dividas": np.array([100.0, 50.0]), "caixa_inicial": 10.0})
        caixa = np.array([[40, 60, 10, 50], [10, 10, 100, 0], [200, 0, 0, 0]], dtype=float)
        saldo, pg = pagamentos(inst, caixa)
        self.assertEqual(pg.tolist(), [[True, True], [False, True], [True, True]])
        self.assertAlmostEqual(saldo[0], 10 + 160 - 150)

    def test_resumir_chaves(self):
        inst = inst_pequena()
        r = resumir(inst, simular(inst, politica_fixa(inst.ref), None, 600, 2, "t"))
        for k in ["p_pagamento", "p_todas", "saldo_liquido_percentil", "cvar", "estoque_final_un", "margem_pct", "risco_ok"]:
            self.assertIn(k, r)
        self.assertEqual(set(r["p_pagamento"]), {4, 8, 12})


class TestDP(unittest.TestCase):
    def _est_deterministico(self, inst):
        dq = np.array([[[8.0], [5.0], [3.0]]])
        z = np.zeros((1, 3))
        return EstatisticasMC(1, dq, dq[:, :, 0], z, np.array([5.0]), z, z)

    def test_dp_igual_a_enumeracao_de_todos_os_caminhos(self):
        inst = inst_minima()
        est = self._est_deterministico(inst)
        for theta in (0.0,):
            sol = resolver_dp(inst, est, theta)
            margem = inst.preco[0] * inst.niveis * (1 - inst.taxa[0]) - inst.custo[0] - 5.0
            melhor = -1e18
            for caminho in itertools.product(range(3), repeat=3):
                s, tot = 6, 0.0
                for w, l in enumerate(caminho):
                    u = min(s, round(est.dq[0, l, 0]))
                    tot += sol.pesos[w] * margem[l] * u
                    s -= u
                melhor = max(melhor, tot)
            self.assertAlmostEqual(sol.valor[0, 0, 6], melhor, places=6)
            cam = reconstruir_caminho(inst, est, sol)             # a trajetória reconstruída atinge o valor ótimo
            self.assertAlmostEqual(float(cam.caixa_esp.sum()), melhor, places=6)

    def test_reuso_de_subproblemas_e_pesos(self):
        inst = inst_minima()
        sol = resolver_dp(inst, self._est_deterministico(inst), 0.0)
        self.assertEqual(sol.estados, 3 * 7)
        self.assertEqual(sol.caminhos_ingenuos, 27.0)
        w = peso_semana(12, (4, 8, 12), 1.0)
        self.assertEqual((w[0], w[5], w[11]), (2.0, 1.5, 1.0))
        np.testing.assert_array_equal(peso_semana(12, (4, 8, 12), 0.0), np.ones(12))

    def test_politica_adaptativa_consulta_tabela(self):
        pol = np.zeros((2, 2, 4), dtype=np.int8); pol[0, 1, 3] = 2; pol[1, 0, 2] = 1
        f = politica_adaptativa(pol)
        np.testing.assert_array_equal(f(0, np.array([[0, 3], [1, 0]])), [[0, 2], [0, 0]])
        np.testing.assert_array_equal(f(1, np.array([[2, 0]])), [[1, 0]])


class TestGreedyQ2(unittest.TestCase):
    def test_orcamento_semanal_respeitado(self):
        inst = inst_pequena()
        est = estimar_estatisticas(inst, 500, 1)
        sol = resolver_dp(inst, est, 0.0)
        cam = reconstruir_caminho(inst, est, sol)
        pl = alocar_promocoes(inst, est, cam, sol)
        self.assertTrue((pl.gasto.sum(axis=1) <= inst.orcamento_sem + 1e-6).all())
        self.assertTrue((pl.lift >= 0).all())
        inst0 = inst_pequena(orcamento_fator=0.0)
        self.assertEqual(alocar_promocoes(inst0, est, cam, sol).gasto.sum(), 0.0)


class TestPipelineQ2(unittest.TestCase):
    def test_execucao_completa_pequena_e_cenario_surpresa(self):
        with pasta_temporaria() as d, pasta_temporaria() as out:
            escrever_instancia(d)
            dados = carregar_dados(d)
            r = executar(dados, Ajustes(), 99001, 5000, 2000, "rapido", out)
            self.assertEqual(r["instancia"]["cenarios_validacao"], 5000)
            self.assertGreaterEqual(r["instancia"]["cenarios_construcao"], 2000)
            for f in ["comparativo_politicas.csv", "prob_pagamento.csv", "distribuicao_caixa_final.png", "trajetoria_preco_estoque.png", "estresse.csv"]:
                self.assertTrue((Path(out) / f).exists(), f)
            r2 = executar(dados, Ajustes(divida_fator=1.5, skus_indisponiveis=("Q2-001",)), 99001, 5000, 2000, "rapido", out)
            self.assertEqual(r2["instancia"]["n_skus"], 3)


if __name__ == "__main__":
    unittest.main()
