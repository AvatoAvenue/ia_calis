import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd

# ==========================
# Cargar dataset
# ==========================

data = pd.read_csv("test_CSV_IA.txt")

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
# Codificar datos
# ==========================

def encode_word(word):
    return [letter2idx[c] for c in word]

def encode_ipa(ipa):
    return [ipa2idx["<sos>"]] + [ipa2idx[c] for c in ipa] + [ipa2idx["<eos>"]]

X = [encode_word(w) for w in words]
Y = [encode_ipa(i) for i in ipas]

max_x = max(len(x) for x in X)
max_y = max(len(y) for y in Y)

def pad(seq, max_len):
    return seq + [0]*(max_len-len(seq))

X = torch.tensor([pad(x,max_x) for x in X])
Y = torch.tensor([pad(y,max_y) for y in Y])

# ==========================
# Modelo Seq2Seq
# ==========================

class G2PModel(nn.Module):

    def __init__(self, input_size, output_size):
        super().__init__()

        self.embedding = nn.Embedding(input_size, 64)
        self.encoder = nn.LSTM(64,128,batch_first=True)

        self.decoder = nn.LSTM(64,128,batch_first=True)
        self.out = nn.Linear(128,output_size)

        self.out_embed = nn.Embedding(output_size,64)

    def forward(self, x, y):

        x = self.embedding(x)

        _, (h,c) = self.encoder(x)

        y = self.out_embed(y)

        out,_ = self.decoder(y,(h,c))

        out = self.out(out)

        return out


model = G2PModel(len(letters), len(ipa_symbols))

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# ==========================
# Entrenamiento
# ==========================

epochs = 300

for epoch in range(epochs):

    optimizer.zero_grad()

    output = model(X, Y[:,:-1])

    loss = criterion(
        output.reshape(-1,output.shape[-1]),
        Y[:,1:].reshape(-1)
    )

    loss.backward()

    optimizer.step()

    if epoch % 50 == 0:
        print("Epoch",epoch,"Loss:",loss.item())

torch.save(model.state_dict(),"g2p_model.pt")

print("Modelo guardado")