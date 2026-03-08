import torch
import torch.nn as nn
import pandas as pd

# ==========================
# Cargar dataset
# ==========================

data = pd.read_csv("test_CSV_IA.txt")
data.columns = data.columns.str.strip()

words = data["text"].str.lower().tolist()
ipas = data["ipa"].tolist()

# ==========================
# Crear vocabularios
# ==========================

letters = sorted(list(set("".join(words))))
ipa_symbols = sorted(list(set("".join(ipas))))

letters = ["<pad>", "<sos>", "<eos>"] + letters
ipa_symbols = ["<pad>", "<sos>", "<eos>"] + ipa_symbols

letter2idx = {c:i for i,c in enumerate(letters)}
ipa2idx = {c:i for i,c in enumerate(ipa_symbols)}
idx2ipa = {i:c for c,i in ipa2idx.items()}

# ==========================
# Modelo
# ==========================

class G2PModel(nn.Module):

    def __init__(self, input_size, output_size):
        super().__init__()

        self.embedding = nn.Embedding(input_size,64)
        self.encoder = nn.LSTM(64,128,batch_first=True)

        self.decoder = nn.LSTM(64,128,batch_first=True)
        self.out = nn.Linear(128,output_size)

        self.out_embed = nn.Embedding(output_size,64)

    def forward(self,x,y):

        x = self.embedding(x)

        _,(h,c) = self.encoder(x)

        y = self.out_embed(y)

        out,_ = self.decoder(y,(h,c))

        out = self.out(out)

        return out

# ==========================
# Cargar modelo entrenado
# ==========================

model = G2PModel(len(letters), len(ipa_symbols))

model.load_state_dict(torch.load("g2p_model.pt"))

model.eval()

print("Modelo cargado correctamente")

##ojpjpjdsfpjjg

max_x = max(len(w) for w in words)

def encode_word(word):
    return [letter2idx[c] for c in word if c in letter2idx]

def pad(seq,max_len):
    return seq + [0]*(max_len-len(seq))

def predict(word):

    x = torch.tensor([pad(encode_word(word),max_x)])

    with torch.no_grad():

        x = model.embedding(x)
        _,(h,c) = model.encoder(x)

        y = torch.tensor([[ipa2idx["<sos>"]]])

        result = ""

        for _ in range(20):

            y_embed = model.out_embed(y)

            out,(h,c) = model.decoder(y_embed,(h,c))

            out = model.out(out[:,-1])

            pred = out.argmax().item()

            if pred == ipa2idx["<eos>"]:
                break

            result += idx2ipa[pred]

            y = torch.tensor([[pred]])

    return result

## probar modelo

#print(predict("database"))
#print(predict("query"))
#print(predict("docker"))
print(predict("requester")) #version correcta: rɪˈkwɛstər
print(predict("stacker")) #version correcta: ˈstækər
print(predict("indexer")) #version correcta: ˈɪndɛksər