import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.distributions as dists
import numpy as np


class MonsterToy:
    def __init__(self, d_model, n_features, feature_prob=0.04, n_monster_features=1):
        self.d_model = d_model
        self.n_monster_features = n_monster_features
        self.n_sparse_features = n_features-n_monster_features
        self.n_features = n_features
        self.feature_prob = feature_prob
    
        bernoulli_probs = torch.ones(self.n_sparse_features)*feature_prob
        self.dist = dists.Bernoulli(bernoulli_probs)

        # random features
        self.features = torch.randn(n_features, d_model)/np.sqrt(d_model)

        self.monster_activities_dist = dists.Beta(10,9)

    def __call__(self, *batch_shape):
        sparse_feature_activities = self.dist.sample(batch_shape)*torch.rand(batch_shape + (self.n_sparse_features,))
        monster_activities_shape = batch_shape + (self.n_monster_features,)
        monster_activities = self.monster_activities_dist.sample(monster_activities_shape)+1.3
        feature_activities = torch.cat((monster_activities, sparse_feature_activities), dim=-1)
        hidden_states = feature_activities @ self.features

        return hidden_states, feature_activities

    @property
    def normed_features(self):
        return self.features.div(torch.norm(self.features, dim=1, keepdim=True))