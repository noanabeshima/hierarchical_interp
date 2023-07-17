from transformers import GPT2LMHeadModel
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import torch.distributions as dists
import numpy as np
from tiktoken import encoding_for_model


enc = encoding_for_model('gpt2')
gpt2 = GPT2LMHeadModel.from_pretrained('gpt2')
embeds = gpt2.transformer.wte.weight.data

toks = np.array([enc.decode([tok_id]).replace(' ', '◦').replace('\n', '⏎').replace('\t', '↦') for tok_id in range(50257)])