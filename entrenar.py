import argparse
import sys
import os
import math
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

from g2p_core import (
    G2PModel, build_vocab, save_vocab,
    encode_word, encode_ipa, pad_seq,
    DEFAULT_DATASET, DEFAULT_MODEL, DEFAULT_VOCAB,
)

# ANSI colors
RESET = "\033[0m"
BOLD  = "\033[1m"
GREEN = "\033[32m"
CYAN  = "\033[36m"
YELLOW = "\033[33m"
RED   = "\033[31m"
DIM   = "\033[2m"

def banner():
    print(f"""
{CYAN}{BOLD}╔══════════════════════════════════════════╗
║        G2P — Entrenamiento del Modelo        ║
╚══════════════════════════════════════════╝{RESET}
""")

def progress_bar(epoch: int, total: int, loss: float, width: int = 30) -> str:
    filled = int(width * epoch / total)
    bar    = "█" * filled + "░" * (width - filled)
    pct    = 100 * epoch / total
    return f"  [{bar}] {pct:5.1f}%  loss: {loss:.4f}"

# ─────────────────────────────────────────────
def load_data(path: str):
    """Carga el CSV, filtra filas sin IPA estándar (marcadas con *)"""
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    df["text"] = df["text"].str.lower().str.strip()
    df["ipa"]  = df["ipa"].str.strip()

    # Filtrar entradas con *
    before = len(df)
    df = df[~df["ipa"].str.endswith("*")]
    dropped = before - len(df)
    if dropped:
        print(f"  {YELLOW}⚠  Se omitieron {dropped} entradas marcadas con '*' "
              f"(acrónimos sin IPA estándar){RESET}")
    return df["text"].tolist(), df["ipa"].tolist()


def build_tensors(words, ipas, vocab):
    max_x = max(len(w) for w in words)
    max_y = max(len(encode_ipa(i, vocab["ipa2idx"])) for i in ipas)

    X = torch.tensor([
        pad_seq(encode_word(w, vocab["letter2idx"]), max_x)
        for w in words
    ])
    Y = torch.tensor([
        pad_seq(encode_ipa(i, vocab["ipa2idx"]), max_y)
        for i in ipas
    ])
    return X, Y, max_x


def train(args):
    banner()
    print(f"  {DIM}Dataset : {args.dataset}")
    print(f"  Épocas  : {args.epochs}")
    print(f"  LR      : {args.lr}")
    print(f"  Batch   : {args.batch}{RESET}\n")

    # Datos
    print(f"  {CYAN}[1/4] Cargando dataset…{RESET}")
    words, ipas = load_data(args.dataset)
    print(f"       {GREEN}✓ {len(words)} pares cargados{RESET}")

    # Vocabulario
    print(f"  {CYAN}[2/4] Construyendo vocabulario…{RESET}")
    vocab = build_vocab(words, ipas)
    save_vocab(vocab, args.vocab)
    print(f"       {GREEN}✓ {len(vocab['letters'])} letras | "
          f"{len(vocab['ipa_symbols'])} símbolos IPA{RESET}")
    print(f"       {GREEN}✓ Vocabulario guardado en '{args.vocab}'{RESET}")

    # Tensores + DataLoader
    print(f"  {CYAN}[3/4] Preparando tensores…{RESET}")
    X, Y, max_x = build_tensors(words, ipas, vocab)
    dataset  = TensorDataset(X, Y)
    loader   = DataLoader(dataset, batch_size=args.batch, shuffle=True)
    print(f"       {GREEN}✓ Shape X: {X.shape} | Y: {Y.shape}{RESET}")

    # Guardar max_x en el vocab para usarlo en inferencia
    import json
    with open(args.vocab, "r", encoding="utf-8") as f:
        vdata = json.load(f)
    vdata["max_src"] = max_x
    with open(args.vocab, "w", encoding="utf-8") as f:
        json.dump(vdata, f, ensure_ascii=False, indent=2)

    # Modelo
    print(f"  {CYAN}[4/4] Iniciando entrenamiento…{RESET}\n")
    model     = G2PModel(len(vocab["letters"]), len(vocab["ipa_symbols"]),
                         embed_dim=args.embed, hidden_dim=args.hidden)
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=30, factor=0.5, min_lr=1e-5
    )

    best_loss = float("inf")
    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        for bx, by in loader:
            optimizer.zero_grad()
            output = model(bx, by[:, :-1])
            loss   = criterion(
                output.reshape(-1, output.shape[-1]),
                by[:, 1:].reshape(-1)
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(loader)
        scheduler.step(avg_loss)

        # Guardar el mejor modelo
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), args.model)

        # Progreso en consola
        if epoch % 10 == 0 or epoch == 1:
            bar = progress_bar(epoch, args.epochs, avg_loss)
            lr_now = optimizer.param_groups[0]["lr"]
            print(f"\r  Época {epoch:>4}/{args.epochs}{bar}  lr={lr_now:.6f}", end="")

    print(f"\n\n  {GREEN}{BOLD}✓ Entrenamiento completo{RESET}")
    print(f"  {GREEN}✓ Mejor loss: {best_loss:.4f}{RESET}")
    print(f"  {GREEN}✓ Modelo guardado en '{args.model}'{RESET}\n")


def parse_args():
    p = argparse.ArgumentParser(description="Entrena el modelo G2P")
    p.add_argument("--dataset", default=DEFAULT_DATASET)
    p.add_argument("--model",   default=DEFAULT_MODEL)
    p.add_argument("--vocab",   default=DEFAULT_VOCAB)
    p.add_argument("--epochs",  type=int,   default=500)
    p.add_argument("--lr",      type=float, default=0.001)
    p.add_argument("--batch",   type=int,   default=32)
    p.add_argument("--embed",   type=int,   default=64)
    p.add_argument("--hidden",  type=int,   default=256)
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
