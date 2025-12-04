# ensembles.py
# Reproduced from DeepMind Grid Cells implementation
import numpy as np
import torch
import torch.nn.functional as F

class CellEnsemble(object):
    """Abstract parent class for place and head direction cell ensembles."""
    def __init__(self, n_cells, soft_targets, soft_init, device='cpu'):
        self.n_cells = n_cells
        self.soft_targets = soft_targets
        # If soft_init is None, default to soft_targets type (logic from original)
        self.soft_init = soft_init if soft_init is not None else soft_targets
        self.device = device

    def get_targets(self, x):
        """Type of target."""
        return self._get_distribution(x, self.soft_targets)

    def get_init(self, x):
        """Type of initialisation."""
        return self._get_distribution(x, self.soft_init)

    def _get_distribution(self, x, dist_type):
        if dist_type == "normalized":
            # TF: targets = tf.exp(self.unnor_logpdf(x))
            return torch.exp(self.unnor_logpdf(x))
        
        lp = self.log_posterior(x)
        
        if dist_type == "softmax":
            return F.softmax(lp, dim=-1)
        elif dist_type == "voronoi":
            # TF: one_hot_max
            idx = torch.argmax(lp, dim=-1)
            return F.one_hot(idx, num_classes=self.n_cells).float()
        elif dist_type == "sample":
            # TF: softmax_sample (Categorical sample)
            # For training targets, softmax is usually preferred over sampling in PyTorch loss
            return F.softmax(lp, dim=-1)
        elif dist_type == "zeros":
            return torch.zeros_like(lp)
        else:
            raise ValueError(f"Unknown distribution type: {dist_type}")

    def log_posterior(self, x):
        # TF: logp - tf.reduce_logsumexp(logp, axis=2, keep_dims=True)
        logp = self.unnor_logpdf(x)
        return F.log_softmax(logp, dim=-1)

    def loss(self, predictions, targets):
        # TF: tf.nn.softmax_cross_entropy_with_logits_v2
        # PyTorch: sum(-target * log_softmax(prediction))
        log_preds = F.log_softmax(predictions, dim=-1)
        loss = -(targets * log_preds).sum(dim=-1).mean()
        return loss

class PlaceCellEnsemble(CellEnsemble):
    def __init__(self, n_cells, stdev=0.35, pos_min=-5, pos_max=5, seed=None,
                 soft_targets="softmax", soft_init="softmax", device='cpu'):
        super(PlaceCellEnsemble, self).__init__(n_cells, soft_targets, soft_init, device)
        # Create a random MoG with fixed cov over the position
        rs = np.random.RandomState(seed)
        self.means = torch.tensor(rs.uniform(pos_min, pos_max, size=(self.n_cells, 2)), 
                                  dtype=torch.float32, device=device)
        self.variances = torch.ones_like(self.means) * stdev**2

    def unnor_logpdf(self, trajs):
        # trajs: [B, T, 2] -> [B, T, 1, 2]
        # means: [N, 2] -> [1, 1, N, 2]
        diff = trajs.unsqueeze(2) - self.means.unsqueeze(0).unsqueeze(0)
        unnor_logp = -0.5 * torch.sum((diff**2) / self.variances.unsqueeze(0).unsqueeze(0), dim=-1)
        return unnor_logp

class HeadDirectionCellEnsemble(CellEnsemble):
    def __init__(self, n_cells, concentration=20, seed=None,
                 soft_targets="softmax", soft_init="softmax", device='cpu'):
        super(HeadDirectionCellEnsemble, self).__init__(n_cells, soft_targets, soft_init, device)
        # Create a random Von Mises with fixed cov
        rs = np.random.RandomState(seed)
        self.means = torch.tensor(rs.uniform(-np.pi, np.pi, (n_cells)), 
                                  dtype=torch.float32, device=device)
        self.kappa = torch.tensor(concentration, dtype=torch.float32, device=device)

    def unnor_logpdf(self, x):
        # x: [B, T, 1]
        return self.kappa * torch.cos(x - self.means.unsqueeze(0).unsqueeze(0))