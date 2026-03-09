import argparse
import os
import csv
import json
from difflib import SequenceMatcher

import torch

from g2p_core import (
    G2PModel, load_vocab, predict as model_predict,
    DEFAULT_DATASET, DEFAULT_MODEL, DEFAULT_VOCAB,
)

RESET  = "\033[0m"
BOLD   = "\033[1m"
DIM    = "\033[2m"
GREEN  = "\033[32m"
CYAN   = "\033[36m"
YELLOW = "\033[33m"
RED    = "\033[31m"

def sim(a, b): return SequenceMatcher(None, a, b).ratio()

def color_sim(s):
    if s >= 0.9: return f"{GREEN}{s:.1%}{RESET}"
    if s >= 0.6: return f"{YELLOW}{s:.1%}{RESET}"
    return f"{RED}{s:.1%}{RESET}"


def evaluate(args):
    # Cargar modelo
    if not os.path.isfile(args.model):
        print(f"  {RED}✗ Modelo no encontrado: {args.model}{RESET}")
        return
    if not os.path.isfile(args.vocab):
        print(f"  {RED}✗ Vocabulario no encontrado: {args.vocab}{RESET}")
        return

    vocab = load_vocab(args.vocab)
    with open(args.vocab, encoding="utf-8") as f:
        vdata = json.load(f)
    max_src = vdata.get("max_src", 30)

    model = G2PModel(len(vocab["letters"]), len(vocab["ipa_symbols"]))
    model.load_state_dict(torch.load(args.model, map_location="cpu"))
    model.eval()

    # Cargar dataset
    rows = []
    with open(args.dataset, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            ipa = row["ipa"].strip()
            if not ipa.endswith("*"):          # solo entradas con IPA real
                rows.append({
                    "text": row["text"].strip().lower(),
                    "ipa":  ipa,
                })

    print(f"""
{CYAN}{BOLD}╔══════════════════════════════════════════╗
║            G2P — Evaluación del Modelo       ║
╚══════════════════════════════════════════╝{RESET}
  Dataset : {args.dataset}  ({len(rows)} entradas válidas)
""")

    # Predicciones
    results = []
    for r in rows:
        pred = model_predict(r["text"], model, vocab, max_src)
        s    = sim(pred, r["ipa"])
        exact = pred == r["ipa"]
        results.append({
            "word":      r["text"],
            "reference": r["ipa"],
            "predicted": pred,
            "similarity": s,
            "exact":     exact,
        })

    # Métricas globales
    n       = len(results)
    exact_n = sum(1 for r in results if r["exact"])
    avg_sim = sum(r["similarity"] for r in results) / n

    print(f"  {BOLD}Resultados globales:{RESET}")
    print(f"   Accuracy exacta   : {GREEN if exact_n/n > 0.5 else YELLOW}"
          f"{exact_n}/{n}  ({exact_n/n:.1%}){RESET}")
    print(f"   Similitud media   : {color_sim(avg_sim)}")
    print()

    # Tabla de peores casos
    worst = sorted(results, key=lambda r: r["similarity"])[:args.top]
    print(f"  {BOLD}Peores {args.top} predicciones:{RESET}\n")
    print(f"  {'PALABRA':<22} {'REFERENCIA':<25} {'PREDICCIÓN':<25} SIM")
    print(f"  {'─'*22} {'─'*25} {'─'*25} {'─'*6}")
    for r in worst:
        ref_short  = r["reference"][:24]
        pred_short = r["predicted"][:24] if r["predicted"] else "(vacío)"
        print(f"  {r['word']:<22} {ref_short:<25} {pred_short:<25} "
              f"{color_sim(r['similarity'])}")

    # Mejores casos
    best = sorted(results, key=lambda r: -r["similarity"])[:args.top]
    print(f"\n  {BOLD}Mejores {args.top} predicciones:{RESET}\n")
    print(f"  {'PALABRA':<22} {'REFERENCIA':<25} {'PREDICCIÓN':<25} SIM")
    print(f"  {'─'*22} {'─'*25} {'─'*25} {'─'*6}")
    for r in best:
        pred_short = r["predicted"][:24] if r["predicted"] else "(vacío)"
        marker = f" {GREEN}✓{RESET}" if r["exact"] else ""
        print(f"  {r['word']:<22} {r['reference']:<25} {pred_short:<25} "
              f"{color_sim(r['similarity'])}{marker}")

    # Exportar
    if args.export:
        out_path = "evaluacion_resultado.csv"
        with open(out_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["word","reference","predicted","similarity","exact"])
            writer.writeheader()
            for r in sorted(results, key=lambda x: x["similarity"]):
                writer.writerow(r)
        print(f"\n  {GREEN}✓ Reporte exportado a '{out_path}'{RESET}")

    print()


def parse_args():
    p = argparse.ArgumentParser(description="Evalúa el modelo G2P")
    p.add_argument("--model",   default=DEFAULT_MODEL)
    p.add_argument("--vocab",   default=DEFAULT_VOCAB)
    p.add_argument("--dataset", default=DEFAULT_DATASET)
    p.add_argument("--top",     type=int, default=15,
                   help="Cuántos mejores/peores mostrar")
    p.add_argument("--export",  action="store_true",
                   help="Exportar resultados a CSV")
    return p.parse_args()


if __name__ == "__main__":
    evaluate(parse_args())
