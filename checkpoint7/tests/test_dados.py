import json
import unittest
from pathlib import Path

from tests.helpers import catalogo_q1, catalogo_q2, config_q1, config_q2, escrever_instancia, pasta_temporaria
from comum.cenarios import Ajustes
from comum.dados import (ErroIntegridade, ErroProvisorio, carregar_dados, sha256_arquivo, validar_catalogo_q1,
                         validar_catalogo_q2, validar_config_q1, validar_config_q2)


class TestDados(unittest.TestCase):
    def test_carga_e_integridade_ok(self):
        with pasta_temporaria() as d:
            escrever_instancia(d)
            dados = carregar_dados(d)
            self.assertTrue(dados.integridade["ok"])
            self.assertEqual(dados.grupo, "G99")
            self.assertEqual(len(dados.q1_catalogo), 6)

    def test_arquivo_alterado_invalida_instancia(self):
        with pasta_temporaria() as d:
            escrever_instancia(d)
            with open(Path(d) / "q1_catalogo_G99.csv", "a") as f:
                f.write("Q1-999,Moda,1,2,3,4,1.0,0.1,1,10\n")
            with self.assertRaises(ErroIntegridade):
                carregar_dados(d)

    def test_arquivo_ausente(self):
        with pasta_temporaria() as d:
            escrever_instancia(d)
            (Path(d) / "q2_config_G99.json").unlink()
            with self.assertRaises(ErroIntegridade):
                carregar_dados(d)

    def test_manifesto_provisorio_exige_flag(self):
        with pasta_temporaria() as d:
            escrever_instancia(d, provisorio=True)
            with self.assertRaises(ErroProvisorio):
                carregar_dados(d)
            self.assertTrue(carregar_dados(d, permitir_provisorio=True).provisorio)

    def test_sem_manifesto(self):
        with pasta_temporaria() as d:
            with self.assertRaises(FileNotFoundError):
                carregar_dados(d)

    def test_entradas_invalidas(self):
        df = catalogo_q1()
        with self.assertRaises(ValueError): validar_catalogo_q1(df.drop(columns=["elasticidade"]))
        with self.assertRaises(ValueError): validar_catalogo_q1(df.assign(estoque=[0] + [5] * 5))
        with self.assertRaises(ValueError): validar_catalogo_q1(df.assign(sku=["A"] * 6))
        with self.assertRaises(ValueError): validar_catalogo_q1(df.assign(taxa_marketplace=[1.2] + [0.1] * 5))
        with self.assertRaises(ValueError): validar_catalogo_q1(df.iloc[0:0])
        with self.assertRaises(ValueError): validar_catalogo_q2(catalogo_q2().assign(preco_concorrente_base=[-1, 1, 1, 1]))
        with self.assertRaises(ValueError): validar_config_q1(config_q1(margem_minima_percentual=1.5))
        with self.assertRaises(ValueError): validar_config_q1(config_q1(niveis_preco_relativos=[]))
        c = config_q2(); c["variabilidade"]["sigma_demanda"] = -1
        with self.assertRaises(ValueError): validar_config_q2(c)

    def test_ajustes_descricao(self):
        self.assertEqual(Ajustes().descricao(), [])
        self.assertIn("meta de caixa x1.1", Ajustes(meta_fator=1.1).descricao())


if __name__ == "__main__":
    unittest.main()
