import numpy as np

from treelib import Node as TreeLibNode
from treelib import Tree as TreeLibTree

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.distributions as dists

from .feature_tree import recursively_add_nodes

def recursively_fill_in_default_values(tree_dict):
    if 'is_read_out' not in tree_dict:
        tree_dict['is_read_out'] = True
    if 'is_allopatry' not in tree_dict:
        tree_dict['is_allopatry'] = False
    if 'is_binary' not in tree_dict:
        tree_dict['is_binary'] = True
    if 'children' in tree_dict:
        for child_dict in tree_dict['children']:
            recursively_fill_in_default_values(child_dict)

class Tree:
    def __init__(self, n_growths=10, root_is_feature=False, tree_dict=False):
        super().__init__()
    

        if tree_dict is not False:
            assert isinstance(tree_dict, dict)
            recursively_fill_in_default_values(tree_dict)
            self.root = Node(tree=self, is_read_out=tree_dict['is_read_out'], is_allopatry=tree_dict['is_allopatry'], is_binary=tree_dict['is_binary'])
        else:
            self.root = Node(tree=self, is_read_out=root_is_feature, is_allopatry=False, is_binary=True)

        self.all_features = [self.root] if self.root.is_read_out else []
        self.all_nodes = [self.root]

        if tree_dict is False:
            for _ in range(n_growths):
                self.root.grow()
        else:
            for child_dict in tree_dict['children']:
                self.root.create_child_from_dict(child_dict)
    def __repr__(self):
        tree_structure = TreeLibTree()
        root_str = ""
        if self.root.is_read_out:
            root_str += "0 "
        if self.root.is_allopatry:
            root_str += "A"
        if self.root.is_binary:
            root_str += "B"
        root_str += " "
        tree_structure.create_node(root_str, 'root')
        # tree_structure.create_node(f'{"0 " if self.root.is_read_out else ""}root', 'root')
        recursively_add_nodes(self.all_features, tree_structure, self.root, 'root')
        # get string representation
        # tree_structure_string.show()
        return str(tree_structure)
    def sample_feature_mask(self, batch_size=1):
        if batch_size == 1:
            for node in self.all_nodes:
                node.is_active = False
            self.root.sample_activation()
            return torch.tensor([node.is_active for node in self.all_features], dtype=torch.float)
        else:
            return torch.stack([self.sample_feature_mask() for _ in range(batch_size)])
    def sample(self, batch_size):
        feature_activation_mask = self.sample_feature_mask(batch_size=batch_size)
        return feature_activation_mask
        # return feature_activation_mask*torch.rand_like(feature_activation_mask)
    @property
    def n_features(self):
        return len(self.all_features)


class Node:
    def __init__(self, tree, is_read_out, is_allopatry, is_binary):
        super().__init__()
        # random name
        self.name = ''.join(np.random.choice(list('abcdefghijklmnopqrstuvwxyz'), 5))
        self.tree = tree
        self.is_read_out = is_read_out
        self.is_allopatry = is_allopatry
        self.is_binary = is_binary
        self.edges = []
        self.is_active = False
    def __repr__(self):
        return f'''Node (
    is_read_out: {self.is_read_out}
    is_allopatry: {self.is_allopatry}
    edges: {self.pretty_child_probs}
)'''
    @property
    def child_probs(self):
        return torch.tensor([edge['child_prob'] for edge in self.edges])
    @property
    def pretty_child_probs(self):
        return ', '.join([f'{edge["child_prob"]:.2f}' for edge in self.edges])
    def __hash__(self):
        return hash(self.name)
    def create_child(self, allopatry_prob=0.25):
        child_is_allopatry = bool(dists.Bernoulli(allopatry_prob).sample())
        # child_is_binary = bool(dists.Bernoulli(0.5).sample()) if child_is_allopatry else bool(dists.Bernoulli(0.4).sample())
        child_is_binary = True
        child_prob_dist = dists.Beta(3, 9)
        # child_prob_dist = dists.Bernoulli(1.0)
        
        if self.is_allopatry:
            child_is_read_out = True
            new_child_probs = torch.cat((self.child_probs, child_prob_dist.sample((1,))))
            new_child_probs = new_child_probs/new_child_probs.sum()
            assert torch.allclose(new_child_probs.sum(), torch.tensor(1.0))
            # print(f'{len(self.edges)=}')
            # print(f'{len(new_child_probs)=}')
            for i, new_child_prob in enumerate(new_child_probs[:-1]):
                self.edges[i]['child_prob'] = new_child_prob
            child_prob = new_child_probs[-1].item()
        else:
            # sample from beta distribution
            child_is_dense = bool(dists.Bernoulli(0.5).sample()) if child_is_allopatry else False
            child_prob = 1.0 if child_is_dense else child_prob_dist.sample().item()
            child_is_read_out = False if child_is_dense else True
            
        child_node = Node(tree=self.tree, is_read_out=child_is_read_out, is_allopatry=child_is_allopatry, is_binary=child_is_binary)
        if child_is_read_out:
            self.tree.all_features.append(child_node)
        self.tree.all_nodes.append(child_node)

        if child_is_allopatry:
            child_node.create_child(allopatry_prob=0.05)
            child_node.create_child(allopatry_prob=0.05)

        self.edges.append({'child_prob': child_prob, 'child': child_node})

        return self.edges[-1]['child']

    def create_child_from_dict(self, child_dict):
        child_is_read_out = child_dict['is_read_out']
        child_is_allopatry = child_dict['is_allopatry']
        child_is_binary = child_dict['is_binary']
        child_prob = child_dict['active_prob']

        child_node = Node(tree=self.tree, is_read_out=child_is_read_out, is_allopatry=child_is_allopatry, is_binary=child_is_binary)
        if child_is_read_out:
            self.tree.all_features.append(child_node)
        self.tree.all_nodes.append(child_node)

        self.edges.append({'child_prob': child_prob, 'child': child_node})

        return self.edges[-1]['child']
    def sample_child(self):
        probs = self.child_probs

        # allopatry_values = [edge['child'].is_allopatry for edge in self.edges]
        # probs = (probs*(1+0.1*torch.tensor(allopatry_values, dtype=torch.float))).clamp(0, 1)

        return self.edges[dists.Categorical(probs).sample()]['child']
    def grow(self):
        local_prob = 0.8 if self.is_allopatry else 0.4
        locally_add_feature = bool(dists.Bernoulli(local_prob).sample()) if len(self.edges) > 0 else True
        if locally_add_feature:
            self.create_child()
        else:
            self.sample_child().grow()
    def sample_activation(self):
        self.is_active = True
        if self.is_allopatry:
            # sample a child
            child = self.sample_child()
            child.sample_activation()
        else:
            for edge in self.edges:
                child_prob, child = edge['child_prob'], edge['child']
                if bool(dists.Bernoulli(child_prob).sample()):
                    child.sample_activation()
    
