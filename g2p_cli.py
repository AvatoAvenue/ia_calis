import argparse
import os
import sys
import csv
import json
import re
import textwrap
from difflib import SequenceMatcher

import torch

from g2p_core import (
    G2PModel, load_vocab, predict as model_predict,
    DEFAULT_DATASET, DEFAULT_MODEL, DEFAULT_VOCAB,
)

# ANSI
RESET  = "\033[0m"
BOLD   = "\033[1m"
DIM    = "\033[2m"
GREEN  = "\033[32m"
CYAN   = "\033[36m"
YELLOW = "\033[33m"
RED    = "\033[31m"
MAGENTA = "\033[35m"
BG_DARK = "\033[48;5;235m"

def c(text, *codes): return "".join(codes) + str(text) + RESET
def ok(msg):    print(f"  {GREEN}✓ {msg}{RESET}")
def warn(msg):  print(f"  {YELLOW}⚠  {msg}{RESET}")
def err(msg):   print(f"  {RED}✗ {msg}{RESET}")
def info(msg):  print(f"  {CYAN}→ {msg}{RESET}")
def sep():      print(f"  {DIM}{'─' * 50}{RESET}")

# Estado global
state = {
    "model":   None,
    "vocab":   None,
    "max_src": 30,
    "dataset": DEFAULT_DATASET,
    "g2p_en":  None,   # lazy-loaded
}

ARPABET_TO_IPA = {
    "AA":"ɑ","AE":"æ","AH":"ʌ","AO":"ɔ","AW":"aʊ","AY":"aɪ",
    "B":"b","CH":"tʃ","D":"d","DH":"ð","EH":"ɛ","ER":"ɝ",
    "EY":"eɪ","F":"f","G":"ɡ","HH":"h","IH":"ɪ","IY":"i",
    "JH":"dʒ","K":"k","L":"l","M":"m","N":"n","NG":"ŋ",
    "OW":"oʊ","OY":"ɔɪ","P":"p","R":"r","S":"s","SH":"ʃ",
    "T":"t","TH":"θ","UH":"ʊ","UW":"u","V":"v","W":"w",
    "Y":"j","Z":"z","ZH":"ʒ",
}


#  Carga del modelo
def load_model(model_path: str, vocab_path: str) -> bool:
    """Carga modelo y vocabulario. Retorna True si tuvo éxito."""
    if not os.path.isfile(model_path):
        warn(f"Modelo no encontrado: '{model_path}'")
        warn("Ejecuta  python entrenar.py  primero.")
        return False
    if not os.path.isfile(vocab_path):
        warn(f"Vocabulario no encontrado: '{vocab_path}'")
        return False

    vocab = load_vocab(vocab_path)
    # Leer max_src guardado durante el entrenamiento
    with open(vocab_path, encoding="utf-8") as f:
        vdata = json.load(f)
    max_src = vdata.get("max_src", 30)

    model = G2PModel(len(vocab["letters"]), len(vocab["ipa_symbols"]))
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()

    state["model"]   = model
    state["vocab"]   = vocab
    state["max_src"] = max_src
    return True


#  g2p_en 
def get_g2p_en():
    if state["g2p_en"] is not None:
        return state["g2p_en"]
    try:
        from g2p_en import G2p
        import nltk
        for corpus in ["averaged_perceptron_tagger_eng", "cmudict"]:
            try: nltk.data.find(corpus)
            except LookupError: nltk.download(corpus, quiet=True)
        state["g2p_en"] = G2p()
        return state["g2p_en"]
    except ImportError:
        return None

def auto_ipa(word: str) -> str | None:
    """Genera IPA usando g2p_en (ARPAbet → IPA)"""
    g2p = get_g2p_en()
    if g2p is None:
        return None
    phonemes = g2p(word)
    result = []
    for p in phonemes:
        if p == " ": continue
        p = re.sub(r"\d", "", p)
        if p in ARPABET_TO_IPA:
            result.append(ARPABET_TO_IPA[p])
    return "".join(result) if result else None


#  Dataset helpers
def read_dataset(path: str) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({
                "text": row["text"].strip(),
                "ipa":  row["ipa"].strip(),
            })
    return rows

def write_dataset(path: str, rows: list[dict]):
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["text", "ipa"])
        writer.writeheader()
        writer.writerows(rows)

def word_in_dataset(word: str, rows: list[dict]) -> dict | None:
    for r in rows:
        if r["text"].lower() == word.lower():
            return r
    return None

def similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, a, b).ratio()


#  Acciones del menú
def action_predict():
    """Predice IPA con el modelo cargado."""
    sep()
    print(c("  PREDECIR IPA", BOLD, CYAN))
    sep()
    if state["model"] is None:
        err("Modelo no cargado. Revisa la ruta o entrena primero.")
        return

    while True:
        raw = input(f"\n  {BOLD}Palabra (Enter para volver):{RESET} ").strip()
        if not raw:
            break
        word = raw.lower()
        ipa  = model_predict(word, state["model"], state["vocab"],
                             state["max_src"])
        print(f"\n  {CYAN}{'─'*30}{RESET}")
        print(f"  Entrada  : {BOLD}{word}{RESET}")
        print(f"  IPA      : {GREEN}{BOLD}{ipa if ipa else '(vacío)'}{RESET}")
        print(f"  {CYAN}{'─'*30}{RESET}")


def action_compare():
    """Compara predicción del modelo vs una IPA de referencia"""
    sep()
    print(c("  COMPARAR: MODELO vs REFERENCIA", BOLD, CYAN))
    sep()
    if state["model"] is None:
        err("Modelo no cargado.")
        return

    rows = read_dataset(state["dataset"])

    while True:
        raw = input(f"\n  {BOLD}Palabra (Enter para volver):{RESET} ").strip()
        if not raw:
            break
        word = raw.lower()

        predicted = model_predict(word, state["model"], state["vocab"],
                                  state["max_src"])

        # Buscar en dataset
        found = word_in_dataset(word, rows)
        ref   = found["ipa"] if found else None

        # IPA automático de g2p_en
        auto  = auto_ipa(word)

        print(f"\n  {CYAN}{'─'*40}{RESET}")
        print(f"  Palabra        : {BOLD}{word}{RESET}")
        print(f"  Modelo G2P     : {GREEN}{BOLD}{predicted or '(vacío)'}{RESET}")
        if ref:
            sim = similarity(predicted, ref) * 100
            bar = "█" * int(sim / 5) + "░" * (20 - int(sim / 5))
            print(f"  Ref. dataset   : {YELLOW}{ref}{RESET}")
            print(f"  Similitud      : [{GREEN}{bar}{RESET}] {sim:.1f}%")
        else:
            print(f"  Ref. dataset   : {DIM}(no encontrada){RESET}")
        if auto:
            print(f"  g2p_en (auto)  : {MAGENTA}{auto}{RESET}")
        print(f"  {CYAN}{'─'*40}{RESET}")

        if not found:
            ans = input(f"  ¿Agregar al dataset con IPA de referencia? [s/n]: ").strip().lower()
            if ans == "s":
                ipa_input = input(f"  IPA correcta (o Enter para usar g2p_en '{auto}'): ").strip()
                final_ipa = ipa_input if ipa_input else auto
                if final_ipa:
                    rows.append({"text": word, "ipa": final_ipa})
                    write_dataset(state["dataset"], rows)
                    ok(f"'{word}' → '{final_ipa}' agregado al dataset.")


def action_add():
    """Agrega una o varias palabras al dataset con IPA a mano."""
    sep()
    print(c("  AGREGAR PALABRAS AL DATASET", BOLD, CYAN))
    sep()
    rows = read_dataset(state["dataset"])
    added = 0

    print(f"  {DIM}Escribe la palabra y su IPA. Deja IPA vacío para usar g2p_en automático.")
    print(f"  Escribe 'listo' como palabra para terminar.{RESET}\n")

    while True:
        word_raw = input(f"  {BOLD}Palabra:{RESET} ").strip()
        if not word_raw or word_raw.lower() == "listo":
            break

        word = word_raw.lower()

        # ¿Ya existe?
        existing = word_in_dataset(word, rows)
        if existing:
            print(f"  {YELLOW}⚠  '{word}' ya existe con IPA: {existing['ipa']}{RESET}")
            upd = input("     ¿Actualizar IPA? [s/n]: ").strip().lower()
            if upd != "s":
                continue

        # Propuesta automática
        auto = auto_ipa(word)
        if auto:
            info(f"IPA automático (g2p_en): {MAGENTA}{auto}{RESET}")

        ipa_raw = input(f"  {BOLD}IPA{RESET} (Enter para usar automático): ").strip()
        final   = ipa_raw if ipa_raw else auto

        if not final:
            warn("No se proporcionó IPA. Entrada omitida.")
            continue

        if existing:
            for r in rows:
                if r["text"] == word:
                    r["ipa"] = final
        else:
            rows.append({"text": word, "ipa": final})

        write_dataset(state["dataset"], rows)
        ok(f"'{word}' → '{final}'")
        added += 1
        print()

    if added:
        ok(f"{added} entrada(s) guardada(s) en '{state['dataset']}'")
        info("Recuerda reentrenar el modelo con:  python entrenar.py")


def action_view():
    """Muestra el dataset paginado."""
    sep()
    print(c("  VER DATASET", BOLD, CYAN))
    sep()
    rows = read_dataset(state["dataset"])
    rows_sorted = sorted(rows, key=lambda r: r["text"])

    page_size = 20
    total     = len(rows_sorted)
    pages     = (total + page_size - 1) // page_size
    page      = 0

    while True:
        start = page * page_size
        chunk = rows_sorted[start:start + page_size]
        print(f"\n  {DIM}Página {page+1}/{pages}  ({total} entradas totales){RESET}\n")
        print(f"  {'#':>4}  {'PALABRA':<25} IPA")
        print(f"  {'─'*4}  {'─'*25} {'─'*30}")
        for i, r in enumerate(chunk, start=start+1):
            ipa_flag = f"{YELLOW}*{RESET}" if r["ipa"].endswith("*") else " "
            print(f"  {i:>4}  {r['text']:<25} {r['ipa']}{ipa_flag}")

        nav = input(f"\n  [n]siguiente [p]anterior [b]uscar [q]salir: ").strip().lower()
        if nav == "n" and page < pages - 1:
            page += 1
        elif nav == "p" and page > 0:
            page -= 1
        elif nav == "b":
            action_search(rows_sorted)
        elif nav == "q":
            break


def action_search(rows=None):
    """Busca una palabra en el dataset."""
    sep()
    print(c("  BUSCAR EN DATASET", BOLD, CYAN))
    sep()
    if rows is None:
        rows = read_dataset(state["dataset"])

    query = input(f"  {BOLD}Buscar:{RESET} ").strip().lower()
    if not query:
        return

    matches = [r for r in rows if query in r["text"]]
    if not matches:
        warn(f"No se encontraron resultados para '{query}'")
        return

    print(f"\n  {GREEN}{len(matches)} resultado(s):{RESET}\n")
    print(f"  {'PALABRA':<25} IPA")
    print(f"  {'─'*25} {'─'*35}")
    for r in matches:
        print(f"  {r['text']:<25} {r['ipa']}")


def action_auto_generate():
    """Genera IPA automático para una lista de palabras y las agrega."""
    sep()
    print(c("  GENERAR IPA AUTOMÁTICO (g2p_en)", BOLD, CYAN))
    sep()
    g2p = get_g2p_en()
    if g2p is None:
        err("g2p_en no está instalado. Ejecuta:  pip install g2p-en")
        return

    print(f"  {DIM}Escribe palabras separadas por coma o una por línea.")
    print(f"  Escribe 'listo' para terminar.{RESET}\n")

    rows = read_dataset(state["dataset"])
    pending = []

    while True:
        line = input(f"  {BOLD}Palabras:{RESET} ").strip()
        if line.lower() == "listo" or not line:
            break
        for w in re.split(r"[,\s]+", line):
            w = w.strip().lower()
            if w:
                pending.append(w)

    if not pending:
        warn("No se ingresaron palabras.")
        return

    print(f"\n  {CYAN}Generando IPA para {len(pending)} palabra(s)…{RESET}\n")
    print(f"  {'PALABRA':<25} {'IPA AUTO':<30} ACCIÓN")
    print(f"  {'─'*25} {'─'*30} {'─'*10}")

    added = 0
    for word in pending:
        auto = auto_ipa(word)
        existing = word_in_dataset(word, rows)
        if existing:
            status = f"{DIM}ya existe{RESET}"
        elif auto:
            rows.append({"text": word, "ipa": auto})
            write_dataset(state["dataset"], rows)
            status = f"{GREEN}agregado{RESET}"
            added += 1
        else:
            status = f"{RED}sin IPA{RESET}"
        ipa_display = auto or "(no generado)"
        print(f"  {word:<25} {ipa_display:<30} {status}")

    print()
    if added:
        ok(f"{added} entrada(s) nueva(s) guardadas.")
        info("Reentrenar con:  python entrenar.py")


def action_stats():
    """Muestra estadísticas del dataset y del modelo."""
    sep()
    print(c("  ESTADÍSTICAS", BOLD, CYAN))
    sep()
    rows = read_dataset(state["dataset"])

    total      = len(rows)
    with_star  = sum(1 for r in rows if r["ipa"].endswith("*"))
    valid      = total - with_star
    avg_word   = sum(len(r["text"]) for r in rows) / max(total, 1)
    avg_ipa    = sum(len(r["ipa"]) for r in rows if not r["ipa"].endswith("*")) / max(valid, 1)
    all_chars  = set("".join(r["text"] for r in rows))
    all_ipa    = set("".join(r["ipa"] for r in rows if not r["ipa"].endswith("*")))

    print(f"\n  {BOLD}Dataset:{RESET}")
    print(f"   Total entradas      : {total}")
    print(f"   Entradas válidas    : {GREEN}{valid}{RESET}")
    print(f"   Entradas con *      : {YELLOW}{with_star}{RESET} (acrónimos/sin IPA)")
    print(f"   Long. media palabra : {avg_word:.1f} chars")
    print(f"   Long. media IPA     : {avg_ipa:.1f} símbolos")
    print(f"   Letras únicas       : {len(all_chars)}")
    print(f"   Símbolos IPA únicos : {len(all_ipa)}")

    print(f"\n  {BOLD}Archivos:{RESET}")
    for path, label in [
        (state["dataset"], "Dataset"),
        (DEFAULT_MODEL,    "Modelo"),
        (DEFAULT_VOCAB,    "Vocabulario"),
    ]:
        if os.path.isfile(path):
            size = os.path.getsize(path)
            print(f"   {GREEN}✓{RESET} {label:<15} {path} ({size:,} bytes)")
        else:
            print(f"   {RED}✗{RESET} {label:<15} {path} {DIM}(no encontrado){RESET}")

    if state["model"] is not None:
        v = state["vocab"]
        print(f"\n  {BOLD}Modelo cargado:{RESET}")
        print(f"   Letras en vocab     : {len(v['letters'])}")
        print(f"   Símbolos IPA vocab  : {len(v['ipa_symbols'])}")
        print(f"   max_src             : {state['max_src']}")
    print()


#  MENÚ PRINCIPAL
MENU = [
    ("1", "Predecir IPA de una palabra",              action_predict),
    ("2", "Comparar modelo vs referencia",            action_compare),
    ("3", "Agregar palabras al dataset",              action_add),
    ("4", "Ver palabras del dataset",                 action_view),
    ("5", "Buscar en el dataset",                     action_search),
    ("6", "Generar IPA automático (g2p_en)",          action_auto_generate),
    ("7", "Estadísticas",                             action_stats),
    ("0", "Salir",                                    None),
]

def print_menu(model_ok: bool):
    os.system("cls" if os.name == "nt" else "clear")
    status = f"{GREEN}●{RESET} Modelo cargado" if model_ok else f"{RED}●{RESET} Sin modelo"
    print(f"""
{CYAN}{BOLD}╔══════════════════════════════════════════╗
║          G2P — Sistema Interactivo           ║
╚══════════════════════════════════════════╝{RESET}
  {status}   {DIM}Dataset: {state['dataset']}{RESET}
""")
    for key, label, _ in MENU:
        dim = DIM if key == "0" else ""
        print(f"  {CYAN}{BOLD}{key}{RESET}  {dim}{label}{RESET}")
    print()


def main(args):
    state["dataset"] = args.dataset

    print(f"\n  {CYAN}Cargando sistema G2P…{RESET}")
    model_ok = load_model(args.model, args.vocab)
    if model_ok:
        ok("Modelo listo")
    else:
        warn("Funciones de predicción no disponibles hasta entrenar el modelo.")

    while True:
        print_menu(model_ok)
        choice = input(f"  {BOLD}Opción:{RESET} ").strip()
        action = next((fn for k, _, fn in MENU if k == choice), None)

        if choice == "0":
            print(f"\n  {DIM}Hasta luego.{RESET}\n")
            sys.exit(0)
        elif action:
            print()
            action()
            input(f"\n  {DIM}[Enter para continuar]{RESET}")
        else:
            warn("Opción inválida.")


def parse_args():
    p = argparse.ArgumentParser(description="CLI interactivo del sistema G2P")
    p.add_argument("--model",   default=DEFAULT_MODEL)
    p.add_argument("--vocab",   default=DEFAULT_VOCAB)
    p.add_argument("--dataset", default=DEFAULT_DATASET)
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
