
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.distributions as dists

from interp_utils import reload_module

import jsonschema
import json
import numpy as np

with open('tree.schema.json', 'r') as f:
    schema = json.load(f)


def validate_tree(instance):
    jsonschema.validate(instance, schema)

with open('tree.json') as f:
    tree_dict = json.load(f)

validate_tree(tree_dict)

class Tree:
    def __init__(self, tree_dict, start_idx=0):
        assert 'active_prob' in tree_dict
        self.active_prob = tree_dict['active_prob']
        self.is_read_out = tree_dict.get('is_read_out', True)
        self.is_allopatry = tree_dict.get('is_allopatry', False)
        self.id = tree_dict.get('id', None)
        
        self.is_binary = tree_dict.get('is_binary', True)
        if self.is_read_out:
            self.index = start_idx
            start_idx += 1
        else:
            self.index = False
    
        self.children = []
        for child_dict in tree_dict.get('children', []):
            child = Tree(child_dict, start_idx)
            start_idx = child.next_index
            self.children.append(child)
        
        self.next_index = start_idx

        if self.is_allopatry:
            assert len(self.children) >= 2
    
    def __repr__(self, indent=0):
        s = ' '*(indent*2)
        s += str(self.index)+' ' if self.index is not False else ' '
        s += 'B' if self.is_binary else ' '
        s += 'A' if self.is_allopatry else ' '
        s += f" {self.active_prob}"

        for child in self.children:
            s += '\n' + child.__repr__(indent+1)
        return s
    
    @property
    def n_features(self):
        return len(self.sample())


    @property
    def child_probs(self):
        return torch.tensor([child.active_prob for child in self.children])

    def sample(self, shape=None, force_inactive=False, force_active=False):
        assert not (force_inactive and force_active)

        # special sampling for shape argument
        if shape is not None:
            if isinstance(shape, int):
                shape = (shape,)
            n_samples = np.prod(shape)
            samples = [self.sample() for _ in range(n_samples)]
            return torch.tensor(samples).view(*shape, -1).float()

        sample = []

        # is this feature active?
        is_active = (torch.rand(1) <= self.active_prob).item()*(1-(force_inactive)) if not force_active else 1     

        # append something if this is a readout
        if self.is_read_out:
            if self.is_binary:
                sample.append(is_active)
            else:
                sample.append((is_active*torch.rand(1)))
        
        if self.is_allopatry:
            active_child = np.random.choice(self.children, p=self.child_probs) if is_active else None
            # if is_active:
            #     print(active_child is not None)

        for child in self.children:
            child_force_inactive  = not bool(is_active) or (self.is_allopatry and child != active_child)
            
            child_force_active = (self.is_allopatry and child == active_child)
            # print(child_force_active)
            sample += child.sample(force_inactive=child_force_inactive, force_active=child_force_active)        

        return sample



