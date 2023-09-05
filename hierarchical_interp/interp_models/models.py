import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.distributions as dists

import transformers

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

import time

from interp_utils import see

class Timer:
    def __init__(self):
        self.last_time = time.time()
    def __call__(self, name):
        new_time = time.time()
        delta = new_time - self.last_time
        print(f'{name}: {delta:.3f} secs')
        self.last_time = new_time

class SparseNNMF(nn.Module):
    def __init__(self, n_features, d_model, orthog_k=0, bias=False, disable_tqdm=False):
        super().__init__()
        assert isinstance(orthog_k, int) or orthog_k is False
        if orthog_k != 0:
            assert orthog_k > 1, 'orthog_k must be > 1'
            self.orthog_mask = nn.Parameter(1-torch.eye(orthog_k), requires_grad=False)
        self.n_features = n_features
        self.d_model = d_model
        self.unsigned_codes = None
        self.atoms = nn.Parameter(torch.randn(n_features, d_model)/np.sqrt(d_model))
        self.orthog_k = orthog_k


        self.bias = nn.Parameter(torch.zeros(d_model)) if bias else None

        self.disable_tqdm = disable_tqdm

        self.norm_atoms()
    def codes(self, codes_subset=None):
        if codes_subset is not None:
            return self.unsigned_codes[codes_subset].abs()
        else:
            return self.unsigned_codes.abs()
    def forward(self, frozen_codes=False, frozen_atoms=False, codes_subset=None):
        if codes_subset is not None:
            codes = self.codes(codes_subset).detach() if frozen_codes else self.codes(codes_subset)
        else:
            codes = self.codes().detach() if frozen_codes else self.codes()
        atoms = self.atoms.detach() if frozen_atoms else self.atoms

        pred = codes @ atoms

        if self.bias is not None:
            pred += self.bias[None]

        return pred, codes
    def norm_atoms(self):
        with torch.no_grad():
            self.atoms.data = F.normalize(self.atoms.data, dim=1)
    
    def orthog_loss(self, codes):
        assert self.orthog_k != 0
        topk_vals, topk_idx = codes.topk(dim=-1, k=self.orthog_k)
        active_atoms = torch.index_select(self.atoms, dim=0, index=topk_idx.view(-1)).view(*topk_idx.shape, -1)
        # see(active_atoms)
        # see(self.orthog_mask)
        orthog_loss = torch.einsum('bkd,bld,kl->bkl', active_atoms, active_atoms, self.orthog_mask).abs().mean()*((self.orthog_k**2)/(self.orthog_k**2-self.orthog_k))
        return orthog_loss

    @property
    def normed_atoms(self):
        return F.normalize(self.atoms, dim=1)
    
    def train(self, train_data, n_epochs=1000, lr=1e-2, sparse_coef = 1e-1, frozen_codes=False, frozen_atoms=False, reinit_codes=False, orthog_coef=0., mean_init=False, alt_sparse_loss=False):

        if self.bias is not None and mean_init is True:
            self.bias.data = train_data.mean(dim=0)
        
        if self.orthog_k != 0 and not frozen_atoms:
            assert orthog_coef > 0, 'orthog_coef must be > 0'
        if reinit_codes or self.unsigned_codes is None or self.unsigned_codes.shape[0] != train_data.shape[0]:

            if self.unsigned_codes is not None and self.unsigned_codes.shape[0] != train_data.shape[0]:
                print('reinitializing codes because train_data size changed')
            self.unsigned_codes = nn.Parameter(torch.randn(train_data.shape[0], self.n_features, device=self.atoms.device)/np.sqrt(self.n_features))
        
        self.alt_sparse_loss = alt_sparse_loss
        
        # if mean_init is True:
        #     mean_dir = train_data.mean(dim=0)
        #     mean_dir = mean_dir/torch.norm(mean_dir)
        #     self.atoms.data[0] = mean_dir
        #     self.unsigned_codes.data[:,0] = train_data @ mean_dir

        
        optimizer = optim.Adam(self.parameters(), lr=lr)
        scheduler = get_scheduler(optimizer, n_epochs)
    
        pbar = tqdm(range(n_epochs)) if not self.disable_tqdm else range(n_epochs)
        for i in pbar:
            pred, codes = self(frozen_codes=frozen_codes, frozen_atoms=frozen_atoms)

            mse_loss = F.mse_loss(pred, train_data.data)

            
            loss = mse_loss

            # with torch.no_grad():
            #     whitened_codes = (codes - codes.mean(dim=-1, keepdim=True))/codes.std(dim=-1, keepdim=True)
            #     whitened_codes = whitened_codes.detach()
            #     atom_corrs = whitened_codes.T @ whitened_codes
            #     atom_corrs = atom_corrs - torch.eye(atom_corrs.shape[0], device=atom_corrs.device)

            # sims = self.atoms @ self.atoms.T
            # sims_loss = (F.relu(sims)*F.relu(-atom_corrs)).mean()
            # loss += 1e-1*sims_loss
                

            if not frozen_codes:
                if self.alt_sparse_loss:
                    with torch.no_grad():
                        per_atom_max = codes.max(dim=0).values[None].detach()
                    sparse_loss = (codes/per_atom_max).mean(dim=-1).mean()
                else:
                    sparse_loss = (codes).mean(dim=-1).mean()
                loss += sparse_coef*sparse_loss
            if self.orthog_k != 0 and not frozen_atoms:
                orthog_loss = self.orthog_loss(codes)
                loss += orthog_coef*orthog_loss
            
            optimizer.zero_grad()
            # see(loss)
            loss.backward()

            optimizer.step()
            scheduler.step()

            if i % 100 == 0:
                loss_string = f'loss: {loss.item():.3f}, mse: {mse_loss.item():.3f}'
                if not frozen_codes:
                    loss_string += f', sparse: {sparse_loss.item():.3f}'
                if self.orthog_k != 0 and not frozen_atoms:
                    loss_string += f', orthog: {orthog_loss.item():.3f}'
                # loss_string += f', lr: {scheduler.get_last_lr()[0]/lr:.3e}'
                
                if not self.disable_tqdm:
                    pbar.set_description(loss_string)

            if not frozen_atoms:
                self.norm_atoms()

class SparseNNMFWCodeBias(nn.Module):
    def __init__(self, n_features, d_model, orthog_k=0, bias=False, disable_tqdm=False):
        super().__init__()
        assert isinstance(orthog_k, int) or orthog_k is False
        if orthog_k != 0:
            assert orthog_k > 1, 'orthog_k must be > 1'
            self.orthog_mask = nn.Parameter(1-torch.eye(orthog_k), requires_grad=False)
        self.n_features = n_features
        self.d_model = d_model
        self.unsigned_precodes = None
        self.atoms = nn.Parameter(torch.randn(n_features, d_model)/np.sqrt(d_model))
        self.orthog_k = orthog_k

        self.unsigned_precode_biases = nn.Parameter(torch.randn(n_features, n_features)/(5*np.sqrt(n_features)))

        self.precode_biases_mask = nn.Parameter(torch.ones(n_features, n_features) - torch.eye(n_features), requires_grad=False)

        self.bias = nn.Parameter(torch.zeros(d_model)) if bias else None

        self.disable_tqdm = disable_tqdm

        self.norm_atoms()
    @property
    def precodes(self):
        return self.unsigned_precodes.abs()
    @property
    def precode_biases(self):
        unnormed_precode_biases = self.unsigned_precode_biases.abs() * self.precode_biases_mask
        precode_biases = unnormed_precode_biases / unnormed_precode_biases.norm(dim=-1, keepdim=True)
        return precode_biases
    def codes_from_precodes(self, precodes):
        # precodes = self.precodes
        precode_biases = self.precode_biases
        return precodes + (precodes @ precode_biases)
    @property
    def codes(self):
        return self.codes_from_precodes(self.precodes)

    def forward(self, frozen_codes=False, frozen_atoms=False):
        precodes = self.precodes
        if frozen_codes:
            precodes = precodes.detach()
        
        codes = self.codes_from_precodes(precodes)
        if frozen_codes:
            codes = codes.detach()

        atoms = self.atoms
        if frozen_atoms:
            atoms = atoms.detach()


        pred = codes @ atoms

        if self.bias is not None:
            pred += self.bias[None]

        return pred, codes, precodes

    def norm_atoms(self):
        with torch.no_grad():
            self.atoms.data = F.normalize(self.atoms.data, dim=1)
    
    def orthog_loss(self, codes):
        assert self.orthog_k != 0
        topk_vals, topk_idx = codes.topk(dim=-1, k=self.orthog_k)
        active_atoms = torch.index_select(self.atoms, dim=0, index=topk_idx.view(-1)).view(*topk_idx.shape, -1)
        orthog_loss = torch.einsum('bkd,bld,kl->bkl', active_atoms, active_atoms, self.orthog_mask).abs().mean()*((self.orthog_k**2)/(self.orthog_k**2-self.orthog_k))
        return orthog_loss

    @property
    def normed_atoms(self):
        return F.normalize(self.atoms, dim=1)
    
    def train(self, train_data, n_epochs=1000, lr=1e-2, sparse_coef = 1e-1, frozen_codes=False, frozen_atoms=False, reinit_codes=False, orthog_coef=0., mean_init=False, alt_sparse_loss=False):

        if self.bias is not None and mean_init is True:
            self.bias.data = train_data.mean(dim=0)
        
        if self.orthog_k != 0 and not frozen_atoms:
            assert orthog_coef > 0, 'orthog_coef must be > 0'
        if reinit_codes or self.unsigned_precodes is None or self.unsigned_precodes.shape[0] != train_data.shape[0]:

            if self.unsigned_precodes is not None and self.unsigned_precodes.shape[0] != train_data.shape[0]:
                print('reinitializing codes because train_data size changed')
            self.unsigned_precodes = nn.Parameter(torch.randn(train_data.shape[0], self.n_features, device=self.atoms.device)/np.sqrt(self.n_features))
        
        
        # if mean_init is True:
        #     mean_dir = train_data.mean(dim=0)
        #     mean_dir = mean_dir/torch.norm(mean_dir)
        #     self.atoms.data[0] = mean_dir
        #     self.unsigned_codes.data[:,0] = train_data @ mean_dir

        
        optimizer = optim.Adam(self.parameters(), lr=lr)
        scheduler = get_scheduler(optimizer, n_epochs)
    
        pbar = tqdm(range(n_epochs)) if not self.disable_tqdm else range(n_epochs)
        for i in pbar:
            pred, codes, precodes = self(frozen_codes=frozen_codes, frozen_atoms=frozen_atoms)

            mse_loss = F.mse_loss(pred, train_data.data)

            
            loss = mse_loss

            if not frozen_codes:

                sparse_loss = (precodes).mean(dim=-1).mean()# + 1e-1*self.precode_biases.mean()
                loss += sparse_coef*sparse_loss
            if self.orthog_k != 0 and not frozen_atoms:
                orthog_loss = self.orthog_loss(codes)
                loss += orthog_coef*orthog_loss
            
            optimizer.zero_grad()
            loss.backward()

            optimizer.step()
            scheduler.step()

            if i % 30 == 0:
                loss_string = f'loss: {loss.item():.3f}, mse: {mse_loss.item():.3f}'
                if not frozen_codes:
                    loss_string += f', sparse: {sparse_loss.item():.3f}'
                if self.orthog_k != 0 and not frozen_atoms:
                    loss_string += f', orthog: {orthog_loss.item():.3f}'
                # loss_string += f', lr: {scheduler.get_last_lr()[0]/lr:.3e}'
                
                if not self.disable_tqdm:
                    pbar.set_description(loss_string)

            if not frozen_atoms:
                self.norm_atoms()

class SparseNNMFWTokIds(nn.Module):
    def __init__(self, n_features, d_model, tok_ids_bias_init, disable_tqdm=False, orthog_k=0):
        super().__init__()
        self.n_features = n_features
        self.d_model = d_model
        self.unsigned_codes = None
        self.atoms = nn.Parameter(torch.randn(n_features, d_model)/np.sqrt(d_model))
        self.bias = nn.Parameter(torch.zeros(d_model))

        self.tok_ids_bias = nn.Embedding.from_pretrained(tok_ids_bias_init)

        self.disable_tqdm = disable_tqdm
        self.orthog_k=orthog_k
        if orthog_k != 0:
            assert orthog_k > 1, 'orthog_k must be > 1'
            self.orthog_mask = nn.Parameter(1-torch.eye(orthog_k), requires_grad=False)

        self.norm_atoms()
    def codes(self, codes_subset=None):
        if codes_subset is not None:
            return self.unsigned_codes[codes_subset].abs()
        else:
            return self.unsigned_codes.abs()
    def forward(self, frozen_codes=False, frozen_atoms=False, codes_subset=None, frozen_bias=False):
        if codes_subset is not None:
            codes = self.codes(codes_subset).detach() if frozen_codes else self.codes(codes_subset)
        else:
            codes = self.codes().detach() if frozen_codes else self.codes()
        atoms = self.atoms.detach() if frozen_atoms else self.atoms

        pred = codes @ atoms

        if self.bias is not None:
            if frozen_bias:
                pred += self.bias[None].detach()
            else:
                pred += self.bias[None]

        return pred, codes
    def norm_atoms(self):
        with torch.no_grad():
            self.atoms.data = F.normalize(self.atoms.data, dim=1)
    
    def orthog_loss(self, codes):
        assert self.orthog_k != 0
        topk_vals, topk_idx = codes.topk(dim=-1, k=self.orthog_k)
        active_atoms = torch.index_select(self.atoms, dim=0, index=topk_idx.view(-1)).view(*topk_idx.shape, -1)
        # see(active_atoms)
        # see(self.orthog_mask)
        orthog_loss = torch.einsum('bkd,bld,kl->bkl', active_atoms, active_atoms, self.orthog_mask).abs().mean()*((self.orthog_k**2)/(self.orthog_k**2-self.orthog_k))
        return orthog_loss

    @property
    def normed_atoms(self):
        return F.normalize(self.atoms, dim=1)
    
    def train(self, train_data, tok_ids, n_epochs=1000, lr=1e-2, sparse_coef = 1e-1, frozen_codes=False, frozen_atoms=False, frozen_bias=False, frozen_tok_id_bias=False, reinit_codes=False, orthog_k=2, orthog_coef=1e-1):

        # if self.bias is not None and mean_init is True:
        #     self.bias.data = train_data.mean(dim=0)
        
        if self.orthog_k != 0 and not frozen_atoms:
            assert orthog_coef > 0, 'orthog_coef must be > 0'
        if reinit_codes or self.unsigned_codes is None or self.unsigned_codes.shape[0] != train_data.shape[0]:

            if self.unsigned_codes is not None and self.unsigned_codes.shape[0] != train_data.shape[0]:
                print('reinitializing codes because train_data size changed')
            self.unsigned_codes = nn.Parameter(torch.randn(train_data.shape[0], self.n_features, device=self.atoms.device)/np.sqrt(self.n_features))
                
        # if mean_init is True:
        #     mean_dir = train_data.mean(dim=0)
        #     mean_dir = mean_dir/torch.norm(mean_dir)
        #     self.atoms.data[0] = mean_dir
        #     self.unsigned_codes.data[:,0] = train_data @ mean_dir

        
        optimizer = optim.Adam(self.parameters(), lr=lr)
        scheduler = get_scheduler(optimizer, n_epochs)
    
        pbar = tqdm(range(n_epochs)) if not self.disable_tqdm else range(n_epochs)
        for i in pbar:
            pred, codes = self(frozen_codes=frozen_codes, frozen_atoms=frozen_atoms, frozen_bias=frozen_bias)

            if frozen_tok_id_bias:
                tok_id_bias = self.tok_ids_bias(tok_ids).detach()
            else:
                tok_id_bias = self.tok_ids_bias(tok_ids)
            mse_loss = F.mse_loss(pred+tok_id_bias, train_data.data)

            
            loss = mse_loss


            if not frozen_codes:
                sparse_loss = (codes).mean(dim=-1).mean()
                loss += sparse_coef*sparse_loss
            if self.orthog_k != 0 and not frozen_atoms:
                orthog_loss = self.orthog_loss(codes)
                loss += orthog_coef*orthog_loss
            
            optimizer.zero_grad()
            loss.backward()

            optimizer.step()
            scheduler.step()

            # if i % 100 == 0:
            #     loss_string = f'loss: {loss.item():.3f}, mse: {mse_loss.item():.3f}'
            #     if not frozen_codes:
            #         loss_string += f', sparse: {sparse_loss.item():.3f}'
            #     # if self.orthog_k != 0 and not frozen_atoms:
            #     #     loss_string += f', orthog: {orthog_loss.item():.3f}'
            #     # loss_string += f', lr: {scheduler.get_last_lr()[0]/lr:.3e}'
                
            #     if not self.disable_tqdm:
            #         pbar.set_description(loss_string)

            if not frozen_atoms:
                self.norm_atoms()


        