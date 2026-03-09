import torch
import torch.nn as nn
import json
import os

#  Rutas por defecto
DEFAULT_DATASET  = "dataset.csv"
DEFAULT_MODEL    = "g2p_model.pt"
DEFAULT_VOCAB    = "g2p_vocab.json"

#  Arquitectura Seq2Seq
class G2PModel(nn.Module):
    def __init__(self, input_size: int, output_size: int,
                 embed_dim: int = 64, hidden_dim: int = 256):
        super().__init__()
        self.embed_dim  = embed_dim
        self.hidden_dim = hidden_dim

        # Encoder
        self.src_embed = nn.Embedding(input_size, embed_dim, padding_idx=0)
        self.encoder   = nn.LSTM(embed_dim, hidden_dim,
                                 num_layers=2, batch_first=True,
                                 dropout=0.3, bidirectional=False)

        # Decoder
        self.tgt_embed = nn.Embedding(output_size, embed_dim, padding_idx=0)
        self.decoder   = nn.LSTM(embed_dim, hidden_dim,
                                 num_layers=2, batch_first=True,
                                 dropout=0.3)

        self.dropout = nn.Dropout(0.3)
        self.out     = nn.Linear(hidden_dim, output_size)

    def forward(self, x, y):
        # x: (batch, src_len)   y: (batch, tgt_len)
        x_emb = self.dropout(self.src_embed(x))
        _, (h, c) = self.encoder(x_emb)

        y_emb = self.dropout(self.tgt_embed(y))
        dec_out, _ = self.decoder(y_emb, (h, c))
        return self.out(dec_out)


#  Gestión de vocabulario
SPECIAL = ["<pad>", "<sos>", "<eos>", "<unk>"]

def build_vocab(words: list[str], ipas: list[str]) -> dict:
    """Construye vocabularios de letras e IPA a partir de listas"""
    letters     = sorted(set("".join(words)))
    ipa_symbols = sorted(set("".join(ipas)))
    letters     = SPECIAL + letters
    ipa_symbols = SPECIAL + ipa_symbols
    return {
        "letters":     letters,
        "ipa_symbols": ipa_symbols,
        "letter2idx":  {c: i for i, c in enumerate(letters)},
        "ipa2idx":     {c: i for i, c in enumerate(ipa_symbols)},
        "idx2ipa":     {i: c for i, c in enumerate(ipa_symbols)},
    }

def save_vocab(vocab: dict, path: str = DEFAULT_VOCAB):
    data = {
        "letters":     vocab["letters"],
        "ipa_symbols": vocab["ipa_symbols"],
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def load_vocab(path: str = DEFAULT_VOCAB) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    letters     = data["letters"]
    ipa_symbols = data["ipa_symbols"]
    return {
        "letters":     letters,
        "ipa_symbols": ipa_symbols,
        "letter2idx":  {c: i for i, c in enumerate(letters)},
        "ipa2idx":     {c: i for i, c in enumerate(ipa_symbols)},
        "idx2ipa":     {i: c for i, c in enumerate(ipa_symbols)},
    }

def vocab_exists(path: str = DEFAULT_VOCAB) -> bool:
    return os.path.isfile(path)

def model_exists(path: str = DEFAULT_MODEL) -> bool:
    return os.path.isfile(path)


#  Codificación / decodificación
def encode_word(word: str, letter2idx: dict) -> list[int]:
    unk = letter2idx.get("<unk>", 3)
    return [letter2idx.get(c, unk) for c in word.lower()]

def encode_ipa(ipa: str, ipa2idx: dict) -> list[int]:
    unk = ipa2idx.get("<unk>", 3)
    seq = [ipa2idx["<sos>"]]
    seq += [ipa2idx.get(c, unk) for c in ipa]
    seq += [ipa2idx["<eos>"]]
    return seq

def pad_seq(seq: list[int], max_len: int) -> list[int]:
    return seq[:max_len] + [0] * max(0, max_len - len(seq))


#  Inferencia
def predict(word: str, model: G2PModel, vocab: dict,
            max_src: int, max_decode: int = 30) -> str:
    """
    Predice la transcripción IPA de una palabra usando greedy decoding.
    Retorna la cadena IPA predicha.
    """
    model.eval()
    letter2idx = vocab["letter2idx"]
    ipa2idx    = vocab["ipa2idx"]
    idx2ipa    = vocab["idx2ipa"]

    enc = pad_seq(encode_word(word, letter2idx), max_src)
    x   = torch.tensor([enc])

    with torch.no_grad():
        x_emb = model.src_embed(x)
        _, (h, c) = model.encoder(x_emb)

        y      = torch.tensor([[ipa2idx["<sos>"]]])
        result = ""

        for _ in range(max_decode):
            y_emb = model.tgt_embed(y)
            dec_out, (h, c) = model.decoder(y_emb, (h, c))
            logits = model.out(dec_out[:, -1])
            pred   = logits.argmax().item()

            if pred == ipa2idx["<eos>"]:
                break
            sym = idx2ipa.get(pred, "")
            if sym not in ("<pad>", "<unk>", "<sos>"):
                result += sym
            y = torch.tensor([[pred]])

    return result
