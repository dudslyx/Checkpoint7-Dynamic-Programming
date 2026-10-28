import unittest

import numpy as np

from tests.helpers import catalogo_q1, config_q1, escrever_instancia, pasta_temporaria
from comum.aleatorio import criar_rng
from comum.cenarios import Ajustes
from comum.dados import carregar_dados
from comum.incentivos import alocar_guloso, fator_lift
from questao_01.brute_force import LimiteBruteForce, brute_force, brute_force_ingenuo, contar_combinacoes
from questao_01.genetic_algorithm import CONFIG_A, CONFIG_B, ConfigGA, executar_ga
from questao_01.greedy import alocar_incentivos
from questao_01.main import executar
from questao_01.modelo import (construir_dominio, construir_instancia, construir_tabelas, avaliar_politica,
                               caixa_sku_ingenuo, dados_ingenuos, escolher_subconjunto, genes_para_niveis, totais)


def montar(n=6, **cfg):
    inst = construir_instancia(catalogo_q1(n), config_q1(**cfg))
    tab = construir_tabelas(inst)
    return inst, tab, construir_dominio(tab)


class TestModelo(unittest.TestCase):
    def test_formulas_manuais(self):
        inst, tab, _ = montar()
        i, l = 2, 3
        p = round(inst.preco[i] * inst.niveis[l], 2)
        d = inst.demanda[i] * (p / inst.preco[i]) ** (-inst.elasticidade[i])
        u = min(inst.estoque[i], d)
        m = p * (1 - inst.taxa[i]) - inst.custo[i] - inst.frete[i]
        self.assertAlmostEqual(tab.caixa[i, l], u * m)
        self.assertAlmostEqual(tab.margem_pct[i, l], m / p)

    def test_piso_de_margem_e_sku_suspenso(self):
        df = catalogo_q1(4)
        df.loc[0, ["custo_unitario"]] = df.loc[0, "preco_atual"] * 2          # custo > preço: nenhum nível viável
        inst = construir_instancia(df, config_q1())
        tab = construir_tabelas(inst)
        dom = construir_dominio(tab)
        self.assertNotIn(0, dom.ativos.tolist())
        self.assertFalse(tab.permitido[0].any())
        for r, i in enumerate(dom.ativos):                                     # todo nível permitido respeita o piso
            for l in dom.niveis_abs[r, :dom.k[r]]:
                self.assertGreaterEqual(tab.margem_pct[i, l], inst.margem_min - 1e-9)
        niveis = genes_para_niveis(dom, np.zeros(len(dom.ativos), dtype=int), inst.n)
        self.assertEqual(niveis[0], -1)
        det = avaliar_politica(inst, tab, niveis)
        self.assertEqual(det.loc[0, "unidades"], 0)

    def test_ajustes_cenario_surpresa(self):
        base = construir_instancia(catalogo_q1(), config_q1())
        aj = construir_instancia(catalogo_q1(), config_q1(), Ajustes(meta_fator=1.1, custo_logistico_fator=1.15, orcamento_fator=0.75,
                                                                       skus_indisponiveis=("Q1-001",)))
        self.assertEqual(aj.n, 5)
        self.assertAlmostEqual(aj.meta, base.meta * 1.1)
        self.assertAlmostEqual(aj.orcamento, base.orcamento * 0.75)
        self.assertAlmostEqual(aj.frete[0], base.frete[1] * 1.15)
        with self.assertRaises(ValueError):
            construir_instancia(catalogo_q1(1), config_q1(), Ajustes(skus_indisponiveis=("Q1-001",)))


class TestBruteForce(unittest.TestCase):
    def test_igual_decomposicao_e_versao_ingenua(self):
        inst, tab, dom = montar(6)
        sub = dom.subconjunto(np.arange(len(dom.k)))
        bf = brute_force(sub.caixa, sub.k, bloco=100)               # bloco pequeno força vários blocos
        self.assertAlmostEqual(bf.melhor_caixa, float(sub.caixa.max(axis=1).sum()))
        self.assertEqual(bf.combinacoes, contar_combinacoes(sub.k))
        perm = [list(sub.niveis_abs[r, :sub.k[r]]) for r in range(len(sub.k))]
        v, _ = brute_force_ingenuo(dados_ingenuos(inst), sub.ativos.tolist(), perm, caixa_sku_ingenuo)
        self.assertAlmostEqual(v, bf.melhor_caixa, places=6)

    def test_casos_limite(self):
        r = brute_force(np.array([[5.0, 9.0]]), np.array([2]))
        self.assertEqual((r.melhor_caixa, r.melhores_genes.tolist(), r.combinacoes), (9.0, [1], 2))
        self.assertEqual(brute_force(np.zeros((0, 1)), np.zeros(0, dtype=int)).combinacoes, 1)
        with self.assertRaises(LimiteBruteForce):
            brute_force(np.zeros((40, 7)), np.full(40, 7))


class TestGA(unittest.TestCase):
    def setUp(self):
        self.inst, self.tab, self.dom = montar(6)

    def test_reprodutivel_por_seed(self):
        a = executar_ga(self.dom.caixa, self.dom.k, CONFIG_A, criar_rng(7, "t"))
        b = executar_ga(self.dom.caixa, self.dom.k, CONFIG_A, criar_rng(7, "t"))
        self.assertEqual(a.melhor_caixa, b.melhor_caixa)
        np.testing.assert_array_equal(a.hist_melhor, b.hist_melhor)

    def test_acha_otimo_do_brute_force_e_nao_o_supera(self):
        bf = brute_force(self.dom.caixa, self.dom.k)
        for cfg in (CONFIG_A, CONFIG_B):
            r = executar_ga(self.dom.caixa, self.dom.k, cfg, criar_rng(3, "t"))
            self.assertLessEqual(r.melhor_caixa, bf.melhor_caixa + 1e-6)
            self.assertAlmostEqual(r.melhor_caixa, bf.melhor_caixa, places=6)

    def test_invariantes(self):
        r = executar_ga(self.dom.caixa, self.dom.k, ConfigGA(populacao=20, geracoes=30, paciencia=None), criar_rng(1, "t"))
        self.assertEqual(r.geracoes_executadas, 30)
        self.assertTrue(np.all(np.diff(r.hist_melhor) >= -1e-9))              # elitismo => melhor nunca piora
        self.assertTrue(np.all((r.melhores_genes >= 0) & (r.melhores_genes < self.dom.k)))   # sempre dentro do domínio permitido
        self.assertEqual(r.avaliacoes, 20 * 30)

    def test_parada_por_paciencia_e_vazio(self):
        r = executar_ga(self.dom.caixa, self.dom.k, ConfigGA(geracoes=500, paciencia=5), criar_rng(1, "t"))
        self.assertLess(r.geracoes_executadas, 500)
        vazio = executar_ga(np.zeros((0, 1)), np.zeros(0, dtype=int), CONFIG_A, criar_rng(1, "t"))
        self.assertEqual(vazio.melhor_caixa, 0.0)


class TestGreedy(unittest.TestCase):
    def test_generico_respeita_orcamento_e_para_sem_ganho(self):
        valor = lambda i, x: 10.0 * (1 - np.exp(-x / 5.0)) * (i + 1)
        x, gasto, lotes = alocar_guloso(3, 12.0, 1.0, valor)
        self.assertLessEqual(gasto, 12.0 + 1e-9)
        self.assertAlmostEqual(x.sum(), gasto)
        self.assertEqual(lotes, int(round(gasto)))
        x0, g0, _ = alocar_guloso(3, 0.0, 1.0, valor)
        self.assertEqual(g0, 0.0)
        x2, g2, _ = alocar_guloso(2, 100.0, 1.0, lambda i, x: 0.5 * x)           # nunca compensa
        self.assertEqual(g2, 0.0)

    def test_integracao_com_politica_do_ga(self):
        inst, tab, dom = montar(6, orcamento_promocional=5000.0)
        ga = executar_ga(dom.caixa, dom.k, CONFIG_A, criar_rng(1, "t"))
        df = avaliar_politica(inst, tab, genes_para_niveis(dom, ga.melhores_genes, inst.n))
        gr = alocar_incentivos(inst, tab, df)
        self.assertLessEqual(gr.gasto_total, inst.orcamento + 1e-9)
        d = gr.detalhe
        self.assertGreaterEqual(d["caixa_depois"].sum(), d["caixa_antes"].sum() - 1e-9)   # só investe com ganho líquido
        self.assertTrue((d["unidades_depois"] <= inst.estoque + 1e-9).all())              # nunca vende mais que o estoque
        sem_folga = (df["estoque_restante"] <= 1e-9).to_numpy()
        self.assertTrue((d["incentivo_rs"].to_numpy()[sem_folga] == 0).all())             # SKU sem estoque livre não recebe incentivo
        inst0, tab0, dom0 = montar(6, orcamento_promocional=0.0)
        gr0 = alocar_incentivos(inst0, tab0, avaliar_politica(inst0, tab0, genes_para_niveis(dom0, ga.melhores_genes, inst0.n)))
        self.assertEqual(gr0.gasto_total, 0.0)

    def test_fator_lift(self):
        self.assertAlmostEqual(float(fator_lift(0.0, 0.3, 10.0)), 1.0)
        self.assertLess(float(fator_lift(1e9, 0.3, 10.0)), 1.3 + 1e-9)
        self.assertEqual(float(fator_lift(5.0, 0.3, 0.0)), 1.0)


class TestPipelineQ1(unittest.TestCase):
    def test_execucao_completa_pequena(self):
        with pasta_temporaria() as d, pasta_temporaria() as out:
            escrever_instancia(d)
            dados = carregar_dados(d)
            r = executar(dados, Ajustes(), 99001, modo="rapido", bf_skus=6, repeticoes=2, saida=out)
            self.assertGreaterEqual(r["meta"]["caixa_final"], r["meta"]["caixa_sem_greedy"] - 1e-6)
            self.assertAlmostEqual(r["ga_validacao"]["gap_medio_A_pct"], 0.0, places=6)
            for f in ["tabela_bf_vs_ga.csv", "convergencia_ga.png", "antes_depois_greedy.csv", "resultado_final.json"]:
                self.assertTrue((__import__("pathlib").Path(out) / f).exists(), f)


if __name__ == "__main__":
    unittest.main()
