"""This is an implementation of the original tmos paper using occhio"""

from occhio.distributions.sparse import SparseUniform
from occhio.autoencoder import AutoEncoder
from occhio.toy_model import ToyModel
import torch
import matplotlib.pyplot as plt

dist = SparseUniform(5, 0.1)
ae = AutoEncoder(5, 2)
tm = ToyModel(dist, ae, importances=torch.tensor([0.8**i for i in range(5)]))

tm.fit(20_000, verbose=True)


W = tm.ae.W.detach().numpy()

plt.scatter(x=W[0], y=W[1])
plt.savefig("test.png")
