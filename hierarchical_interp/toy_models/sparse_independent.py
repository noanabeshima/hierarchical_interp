import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.distributions as dists
import numpy as np


class SparseIndependent:
    def __init__(self, n_features, d_model, feature_sparsity=0.04):
        super().__init__()
        self.n_features = n_features
        self.feature_sparsity = feature_sparsity
    
        self.d_model = d_model
    
        # random features
        self.features = torch.randn(n_features, d_model)/np.sqrt(d_model)
    
    @property
    def normed_features(self):
        return self.features / torch.norm(self.features, dim=-1, keepdim=True)

    def __call__(self, *shape):
        active_features = (torch.rand(*shape, self.n_features) < self.feature_sparsity)
        features = active_features * torch.rand(*shape, self.n_features)
        hidden_state = features @ self.features
        return hidden_state, features

feature_dist = dists.LogNormal(0.3, 0.8)
class ScaledSparseIndependent:
    def __init__(self, n_features, d_model, feature_sparsity=0.04):
        super().__init__()
        self.n_features = n_features
        self.feature_sparsity = feature_sparsity
    
        self.d_model = d_model
    
        # random features
        feature_scales = feature_dist.sample((n_features,))
        self.features = torch.randn(n_features, d_model)
        self.features = F.normalize(self.features, dim=-1) * feature_scales[:, None]

    
    @property
    def normed_features(self):
        return self.features / torch.norm(self.features, dim=-1, keepdim=True)

    def __call__(self, *shape):
        active_features = (torch.rand(*shape, self.n_features) < self.feature_sparsity)
        features = active_features * torch.rand(*shape, self.n_features)
        hidden_state = features @ self.features
        return hidden_state, features

