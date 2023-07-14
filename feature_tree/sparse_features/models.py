import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.distributions as dists

class Autoencoder(nn.Module):
    def __init__(self, n_features, d_model):
        super().__init__()
        self.n_features = n_features
        self.d_model = d_model
        self.encoder = nn.Linear(n_features, d_model)
        self.decoder = nn.Linear(d_model, n_features)
    def forward(self, x):
        hidden_state = self.encoder(x)
        return F.gelu(self.decoder(hidden_state))
    

class SparseAutoencoder(nn.Module):
    def __init__(self, n_features, d_model):
        super().__init__()
        self.n_features = n_features
        self.d_model = d_model
        self.encoder = nn.Linear(d_model, n_features)
        self.decoder = nn.Linear(n_features, d_model)
    def forward(self, x):
        preacts = self.encoder(x)
        acts = F.gelu(preacts)
        return self.decoder(acts), acts
    def norm_atoms(self):
        with torch.no_grad():
            self.decoder.weight.div_(torch.norm(self.decoder.weight, dim=1, keepdim=True))

class SparseNNMF(nn.Module):
    def __init__(self, n_features, d_model, n_codes):
        super().__init__()
        self.n_features = n_features
        self.d_model = d_model
        self.unsigned_codes = nn.Parameter(torch.randn(n_codes, n_features))
        self.atoms = nn.Parameter(torch.randn(n_features, d_model))
    @property
    def codes(self):
        return self.unsigned_codes.abs()
    def forward(self):
        return (self.codes @ self.atoms), self.codes
    def norm_atoms(self):
        with torch.no_grad():
            self.atoms.div_(torch.norm(self.atoms, dim=1, keepdim=True))