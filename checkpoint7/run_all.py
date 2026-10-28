"""Comando único: verifica integridade -> Questão 1 -> Questão 2 -> README/docs.

    python run_all.py                 # dados oficiais em data/
    python run_all.py --seed 1020     # seed explícita (padrão: seed_final do manifesto)
    python run_all.py --rapido        # sem benchmarks (smoke test)
"""
import argparse
import subprocess
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parent


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--rapido", action="store_true")
    p.add_argument("--permitir-provisorio", action="store_true")
    a, extra = p.parse_known_args()
    comum = (["--seed", str(a.seed)] if a.seed is not None else []) + (["--permitir-provisorio"] if a.permitir_provisorio else []) + extra
    modo = ["--modo", "rapido" if a.rapido else "completo"]
    for q in ("questao_01", "questao_02"):
        r = subprocess.run([sys.executable, str(RAIZ / q / "main.py")] + modo + comum)
        if r.returncode:
            sys.exit(r.returncode)
    subprocess.run([sys.executable, str(RAIZ / "scripts" / "gerar_docs.py")], check=True)


if __name__ == "__main__":
    main()
