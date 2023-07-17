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
    @property
    def write_out(self):
        return self.encoder.weight.data.T
    @property
    def write_out_b(self):
        return self.encoder.bias.data
    @property
    def read_in(self):
        return self.decoder.weight.data
    @property
    def read_in_b(self):
        return self.decoder.bias.data
    
    

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

from interp_utils import get_scheduler
from tqdm import tqdm
import numpy as np

class SparseNNMF(nn.Module):
    def __init__(self, n_features, d_model):
        super().__init__()
        self.n_features = n_features
        self.d_model = d_model
        # self.unsigned_codes = nn.Parameter(torch.randn(n_codes, n_features)/np.sqrt(n_features))
        self.unsigned_codes = None
        self.atoms = nn.Parameter(torch.randn(n_features, d_model)/np.sqrt(d_model))
        self.norm_atoms()
    @property
    def codes(self):
        return self.unsigned_codes.abs()
    def forward(self, frozen_codes=False, frozen_atoms=False):
        codes = self.codes.detach() if frozen_codes else self.codes
        atoms = self.atoms.detach() if frozen_atoms else self.atoms

        return (codes @ atoms), codes
    def norm_atoms(self):
        with torch.no_grad():
            self.atoms.data = F.normalize(self.atoms.data, dim=1)
    
    def train(self, batch, n_steps=1000, lr=1e-2, l1_lambda = 1e-1, frozen_codes=False, frozen_atoms=False, reinit_codes=False):
        if reinit_codes or self.unsigned_codes is None or self.unsigned_codes.shape[0] != batch.shape[0]:
            if self.unsigned_codes is not None and self.unsigned_codes.shape[0] != batch.shape[0]:
                print('reinitializing codes because batch size changed')
            self.unsigned_codes = nn.Parameter(torch.randn(batch.shape[0], self.n_features)/np.sqrt(self.n_features))
        
        optimizer = optim.Adam(self.parameters(), lr=lr)
        scheduler = get_scheduler(optimizer, n_steps)
    
        pbar = tqdm(range(n_steps))
        for i in pbar:
            pred, codes = self(frozen_codes=frozen_codes, frozen_atoms=frozen_atoms)
            mse_loss = F.mse_loss(pred, batch.data)
            if frozen_codes:
                loss = mse_loss
            else:
                sparse_loss = codes.abs().mean(dim=-1).mean()
                loss = mse_loss + l1_lambda*sparse_loss



            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            if frozen_codes:
                pbar.set_description(f'loss: {loss.item():.3f}, mse: {mse_loss.item():.3f}')
            else:
                pbar.set_description(f'loss: {loss.item():.3f}, mse: {mse_loss.item():.3f}, sparse: {sparse_loss.item():.3f}')
            self.norm_atoms()


        