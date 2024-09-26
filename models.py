import torch.nn.functional as F
import torch.nn as nn
import torch
import numpy as np
from noa_tools import get_wsd_scheduler

def get_pareto_block_bounds(n_features, n_blocks, min_features):
    if n_blocks == 1:
        return torch.tensor([0, n_features])
    pareto_cdf = 1 - (torch.arange(n_features - min_features + 1)/(n_features - min_features + 1))**(.5)
    x = pareto_cdf

    scaled_pdf = np.concatenate([np.zeros(min_features), x[1:] - x[:-1]], axis=0)
    pdf = scaled_pdf/scaled_pdf.sum()
    
    split_points = np.random.choice(n_features, size=n_blocks-1, replace=False, p=pdf)
    split_points.sort()
    split_points = np.concatenate([[0], split_points, [n_features]])
    return torch.tensor(split_points)


class RunningAvgNormalizer(nn.Module):
    def __init__(self, alpha=0.99):
        super().__init__()
        self.running_avg = nn.Parameter(torch.zeros(1), requires_grad=False)
        self.alpha = alpha

    @torch.no_grad()
    def normalize(self, x, update=False):
        if update is True:
            with torch.no_grad():
                if self.running_avg is None:
                    self.running_avg.data = x.norm(dim=-1).mean()
                else:
                    self.running_avg.data = self.alpha*self.running_avg + (1-self.alpha)*x.norm(dim=-1).mean()
        return x*(np.sqrt(x.shape[-1])/self.running_avg.detach())

    @torch.no_grad()
    def unnormalize(self, x):
        return x*(self.running_avg.detach()/np.sqrt(x.shape[-1]))
        


class L1RegController(nn.Module):
    def __init__(self, eps=0.0003, target_l0=100):
        super().__init__()
        self.l1_scale = 1
        self.eps = eps
        self.target_l0 = target_l0

    @torch.no_grad()
    def forward(self, avg_l0, target_l0=None):
        target_l0 = self.target_l0 if target_l0 is None else target_l0
        if avg_l0 < target_l0:
            self.l1_scale *= (1 - self.eps)
        elif avg_l0 >= target_l0:
            self.l1_scale *= (1 + self.eps)
        return self.l1_scale


        

class SegmentedSAE(nn.Module):
    def __init__(self, d_model, n_features, n_blocks, min_features, target_l0, **config):
        super().__init__()
        self.W_enc = nn.Parameter(torch.randn(d_model, n_features)/(np.sqrt(d_model)))
        self.b_enc = nn.Parameter(-0.04*torch.ones(n_features))
        self.W_dec = nn.Parameter((0.1*self.W_enc.data/self.W_enc.data.norm(dim=0, keepdim=True)).T)
        self.b_dec = nn.Parameter(torch.zeros(d_model,))
        
        self.n_features = n_features
        self.d_model = d_model
        self.n_blocks = n_blocks
        self.min_features = min_features
        
        self.normalizer = RunningAvgNormalizer()
        self.l1_controller = L1RegController(target_l0=target_l0)
        
        
        self.optimizer = torch.optim.Adam(self.parameters(), lr=config['lr'], betas=(config['adam_beta1'], config['adam_beta2']))
        self.scaler = torch.cuda.amp.GradScaler()
        self.scheduler = get_wsd_scheduler(self.optimizer, n_steps=config['n_steps'], n_warmup_steps=50, percent_cooldown=0.2, end_lr_factor=0.1)

        self.config = config
        
        block_weights = torch.arange(1,n_blocks+1)**config['block_weight_power']
        self.block_weights = nn.Parameter(block_weights/block_weights.max(), requires_grad=False)
        self.total_weight_per_block = nn.Parameter(torch.cumsum(self.block_weights.flip(0), dim=0).flip(0), requires_grad=False)
    
    @torch.no_grad()
    def get_acts(self, x, indices=None):
        x = self.normalizer.normalize(x, update=False)
        if indices is None:
            preacts = x @ self.W_enc + self.b_enc
        else:
            preacts = x @ self.W_enc[:,indices] + self.b_enc[indices]
        return F.relu(preacts)*self.W_dec.norm(dim=1)[None]

    def step(self, x, return_metrics=False):
        # self.norm_decoder()
        
        x = self.normalizer.normalize(x, update=True)
        
        split_points = get_pareto_block_bounds(self.n_features, self.n_blocks, self.min_features).to(self.W_enc.device)

        # acts = [F.relu(x @ self.W_enc[:,block_start:block_end] + self.b_enc[block_start:block_end]) for block_start, block_end in zip(split_points[:-1], split_points[1:]) ]
        
        # Get the norms of W_dec
        W_dec_norms = self.W_dec.norm(dim=1)

        # Perform one big matrix multiplication
        all_acts = F.relu(x @ self.W_enc + self.b_enc)
        
        ### get feat perm
        # Calculate average magnitude of all_acts * W_dec_norms
        avg_magnitudes = (all_acts * W_dec_norms[None, :]).mean(dim=0)
    
        # Get permutation indices based on descending order of magnitudes
        feat_perm = torch.argsort(avg_magnitudes, descending=True)


        # Apply sparsity weights to the feature permutation
        weighted_magnitudes = avg_magnitudes * sparsity_weights
        feat_perm = torch.argsort(weighted_magnitudes, descending=True)

        # avg_l0s = (all_acts > 0).float().mean(dim=0)
        # feat_perm = torch.argsort(avg_l0s, descending=True)
        ###

        
        # Split the result into blocks
        acts = [all_acts[:, block_start:block_end] for block_start, block_end in zip(split_points[:-1], split_points[1:])]
        

        # block_l1_losses = [(torch.log(block_acts*W_dec_norms[block_start:block_end][None]+(0.1))).sum(dim=-1).mean() for block_acts, block_start, block_end in zip(acts, split_points[:-1], split_points[1:])]
        block_l1_losses = []
        for block_acts, block_start, block_end in zip(acts, split_points[:-1], split_points[1:]):
            normed_block_acts = block_acts*W_dec_norms[block_start:block_end][None]
            log_loss = (torch.log(normed_block_acts+0.1)-np.log(0.1)).sum(dim=-1).mean()
            # x_loss = 0.1*(normed_block_acts).sum(dim=-1).mean()
            # block_l1_losses.append(log_loss/(log_loss.abs().detach()+1e-3))
            block_l1_losses.append(log_loss)

        preds = [block_acts @ self.W_dec[block_start:block_end] for block_acts, block_start, block_end in zip(acts, split_points[:-1], split_points[1:])]

        with torch.no_grad():
            avg_l0 = np.sum([(block_acts > 0).float().sum().cpu() for block_acts in acts])/acts[0].shape[0]
            l1_reg_scale = self.l1_controller(avg_l0)
            
        l1_loss = ((self.block_weights)*torch.cumsum(torch.stack(block_l1_losses), dim=0)).mean()
        
        preds[0] = preds[0] + self.b_dec
        block_preds = torch.cumsum(torch.stack(preds), dim=0)

        # mse_loss = ((block_preds - x[None])**2).sum(dim=-1).mean()
        block_errs = ((block_preds - x[None])**2).sum(dim=-1).mean(dim=-1)
        # block_errs=block_errs/block_errs.detach()
        block_errs = block_errs*(self.block_weights)
        mse_loss = block_errs.mean()
        
        
        loss = mse_loss + l1_reg_scale*l1_loss
        
        result = {'loss': loss, 'avg_l0': avg_l0, 'l1_reg_scale': l1_reg_scale}
        if return_metrics:
            with torch.no_grad():
                tot_var = ((x-x.mean(dim=0, keepdim=True))**2).sum(dim=-1).mean()
                # block preds: (block, batch, d_model)
                block_errs = ((block_preds - x[None])**2).sum(dim=-1).mean(dim=-1)
                for i in range(block_preds.shape[0]):
                    result[f'block_{i}_fvu'] = block_errs[i]/tot_var
                result['last_block_fvu'] = block_errs[-1]/tot_var
                fvu = (mse_loss/tot_var)
                result['avg_fvu'] = fvu
        
        self.optimizer.zero_grad()
        self.scaler.scale(loss).backward()
        
        ### Scale gradients per-latent by number of blocks it's a part of in the batch ###
        # Calculate the number of split points to the right of each latent
        # num_splits_right = torch.zeros(self.n_features).to(self.W_enc.device).to(torch.int)
        # for i in range(1, len(split_points)):
        #     num_splits_right[split_points[i-1]:split_points[i]] = len(split_points) - i

        # assert num_splits_right.min() >= 1
        # assert num_splits_right.max() <= 10
        # self.W_enc.grad /= num_splits_right[None]
        # self.b_enc.grad /= num_splits_right
        # self.W_dec.grad /= num_splits_right[:,None]

        # Use block weights instead of num_splits_right
        feat_weights = torch.zeros(self.n_features).to(self.W_enc.device)
        for i in range(len(split_points)-1):
            feat_weights[split_points[i]:split_points[i+1]] = self.total_weight_per_block[i]

        self.W_enc.grad /= feat_weights[None]
        self.b_enc.grad /= feat_weights
        self.W_dec.grad /= feat_weights[:,None]
        #########

        
        self.scaler.unscale_(self.optimizer)
        
        # Clip gradients
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
        
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.scheduler.step()

        # # Permute W_dec, W_enc, and b_enc using the previously defined permutation
        # with torch.no_grad():
        #     self.W_dec.data = self.W_dec.data[feat_perm]
        #     self.W_enc.data = self.W_enc.data[:, feat_perm]
        #     self.b_enc.data = self.b_enc.data[feat_perm]
    
        # # Permute optimizer state for W_dec, W_enc, and b_enc
        # for param_group in self.optimizer.param_groups:
        #     for param in param_group['params']:
        #         if param is self.W_dec and param in param_group.get('state', []):
        #             state = param_group.get('state', [])[param]
        #             if 'exp_avg' in state:
        #                 state['exp_avg'] = state['exp_avg'][feat_perm]
        #             if 'exp_avg_sq' in state:
        #                 state['exp_avg_sq'] = state['exp_avg_sq'][feat_perm]
        #         elif param is self.W_enc and param in param_group.get('state', []):
        #             state = param_group.get('state', [])[param]
        #             if 'exp_avg' in state:
        #                 state['exp_avg'] = state['exp_avg'][:, feat_perm]
        #             if 'exp_avg_sq' in state:
        #                 state['exp_avg_sq'] = state['exp_avg_sq'][:, feat_perm]
        #         elif param is self.b_enc and param in param_group.get('state', []):
        #             state = param_group.get('state', [])[param]
        #             if 'exp_avg' in state:
        #                 state['exp_avg'] = state['exp_avg'][feat_perm]
        #             if 'exp_avg_sq' in state:
        #                 state['exp_avg_sq'] = state['exp_avg_sq'][feat_perm]
        
        return result
    
    def norm_decoder(self):
        with torch.no_grad():
            self.W_dec.data = self.W_dec.data/self.W_dec.data.norm(dim=-1, keepdim=True)


# mlp = SplitMLP(100, 25000, 10, 10)

# mlp(torch.randn(4, 100), return_metrics=True)
        