from pathlib import Path
import requests

DATA_PATH = Path("data")
PATH = DATA_PATH / "mnist"

PATH.mkdir(parents=True, exist_ok=True)

URL = "https://github.com/pytorch/tutorials/raw/main/_static/"
FILENAME = "mnist.pkl.gz"

if not (PATH / FILENAME).exists():
        content = requests.get(URL + FILENAME).content
        (PATH / FILENAME).open("wb").write(content)
        
import pickle
import gzip

with gzip.open((PATH / FILENAME).as_posix(), "rb") as f:
        ((x_train, y_train), (x_valid, y_valid), _) = pickle.load(f, encoding="latin-1")

import torch

from torch.utils.data import TensorDataset, DataLoader

epochs = 10

x_train, y_train, x_valid, y_valid = map(
    torch.tensor, (x_train, y_train, x_valid, y_valid)
)

train_ds = TensorDataset(x_train, y_train)
print(train_ds.__len__())
train_dl = DataLoader(train_ds, batch_size=64, shuffle=True)


valid_ds = TensorDataset(x_valid, y_valid)
valid_dl = DataLoader(valid_ds, batch_size=64 * 2)


import math
import torch.nn.functional as F

loss_func = F.cross_entropy

def accuracy(out, b):
    preds = torch.argmax(out, dim=1)
    return (preds==b).float().mean()

# print(accuracy(model(xb), yb))

from torch import nn

class Mnist_Logistic(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(784,10)
    def forward(self, xb):
        return self.lin(xb)
    
model = Mnist_Logistic()

from torch import optim
import numpy as np

opt = optim.SGD(model.parameters(), lr=1e-3)

def loss_batch(model, loss_func, xb, yb, opt=None):
    loss = loss_func(model(xb), yb)
    
    if opt is not None:
        loss.backward()
        opt.step()
        opt.zero_grad()
    
    return loss.item(), len(xb)

def fit(epochs, model, loss_func, opt, train_dl, valid_dl):
    for epoch in range(epochs):
        model.train()
        for xb, yb in train_dl:
            loss_batch(model=model, loss_func=loss_func, xb=xb, yb=yb, opt=opt)
        
        model.eval()
        with torch.no_grad():
            losses, nums = zip(*[loss_batch(model, loss_func, xb, yb) for xb, yb in valid_dl])
        val_loss = np.sum(np.multiply(losses, nums))/np.sum(nums)
        
        print(epoch, val_loss)

    # print(epoch, valid_loss / len(v   alid_dl))
            
fit(epochs, model, loss_func, opt, train_dl, valid_dl)
    