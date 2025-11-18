import torch
from torch import Tensor, nn
import numpy as np

class LatentEDM(nn.Module):
    def __init__(self,
                 model,
                 diffusion_net,
                 sample_std,
                 t,
                 ):
        super().__init__()
        self.sigma_min = 0.01
        self.autoencoder = model
        self.model = diffusion_net
        self.sample_std = sample_std
        self.t = t


    def precond(self, x, t, class_labels, **model_kwargs):
        x = x.to(torch.float32)
        t = self._append_dims(t, x.ndim).to(torch.float32)
        class_labels = None
        pred = self.model(x, t.flatten(), class_labels=class_labels)
        return pred

    def loss(self, x, step, labels=None):
        log_dict = {}
        batch_size = x.size(0)
        device = x.device
        t = self._append_dims(torch.rand(batch_size, device=device), x.ndim)
        noise = torch.randn_like(x)
        x_t = (1 - (1 - self.sigma_min) * t) * noise + t * x
        pred = self.precond(x_t, t, labels)
        u_t = (x - (1 - self.sigma_min) * x_t) / (1 - (1 - self.sigma_min) * t)
        fm_loss = (pred - u_t).pow(2).reshape(batch_size, -1).sum(1).mean()
        return fm_loss, log_dict, None

    @torch.no_grad()
    def sample(self, sample_shape, n_iters, device, class_labels=None, idx=None, temperature=1):
        x_t = torch.randn(sample_shape, device=device)
        time_steps = torch.linspace(0., 1., n_iters)
        delta_t = time_steps[1] - time_steps[0]
        for time_step in time_steps[1:]:
            t = torch.ones(sample_shape[0], device=device) * time_step
            pred = self.precond(x_t, t, None)
            x_t = x_t + pred * delta_t
        t = torch.ones(x_t.shape[0], dtype=torch.float32, device=device) * self.t
        self.autoencoder.eval()
        x_next = self.autoencoder.decode(x_t * self.sample_std, t, None)[0]
        return x_next