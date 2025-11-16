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
        self.sigma_data = 0.5
        self.P_std = 1.2
        self.P_mean = -1.2
        self.model = model
        self.diffusion_net = diffusion_net
        self.sample_std = sample_std
        self.t = t


    def precond(self, x, t, class_labels, **model_kwargs):
        x = x.to(torch.float32)
        sigma = t.to(torch.float32).reshape(-1, 1, 1, 1)
        class_labels = None
        dtype = torch.float32

        c_skip = self.sigma_data ** 2 / (sigma ** 2 + self.sigma_data ** 2)
        c_out = sigma * self.sigma_data / (sigma ** 2 + self.sigma_data ** 2).sqrt()
        c_in = 1 / (self.sigma_data ** 2 + sigma ** 2).sqrt()
        c_noise = sigma.log() / 4

        F_x = self.model((c_in * x).to(dtype), c_noise.flatten(), class_labels=class_labels, **model_kwargs)
        assert F_x.dtype == dtype
        D_x = c_skip * x + c_out * F_x.to(torch.float32)
        return D_x

    def loss(self, x, step, labels=None):
        log_dict = {}
        rnd_normal = torch.randn([x.shape[0], 1, 1, 1], device=x.device)
        sigma = (rnd_normal * self.P_std + self.P_mean).exp()
        weight = (sigma ** 2 + self.sigma_data ** 2) / (sigma * self.sigma_data) ** 2
        y = x
        augment_labels = None
        n = torch.randn_like(y) * sigma
        D_yn = self.diffusion_net(y + n, sigma, labels, augment_labels=augment_labels)
        loss = weight * ((D_yn - y) ** 2)
        return loss, log_dict, None

    @torch.no_grad()
    def sample(self, sample_shape, n_iters, device, class_labels=None, idx=None, temperature=1):
        # Adjust noise levels based on what's supported by the network.
        sigma_min = 0.002
        sigma_max = 80
        num_steps = 36
        rho = 7
        S_churn = 0
        S_min = 0
        S_max = float('inf')
        S_noise = 1
        # Time step discretization.
        step_indices = torch.arange(num_steps, dtype=torch.float64, device=device)
        t_steps = (sigma_max ** (1 / rho) + step_indices / (num_steps - 1) * (
                    sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
        t_steps = torch.cat([torch.as_tensor(t_steps), torch.zeros_like(t_steps[:1])])  # t_N = 0

        latents = torch.randn(sample_shape, device=device) * sigma_max
        # Main sampling loop.
        x_next = latents.to(torch.float64) * t_steps[0]
        for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):  # 0, ..., N-1
            x_cur = x_next

            # Increase noise temporarily.
            gamma = min(S_churn / num_steps, np.sqrt(2) - 1) if S_min <= t_cur <= S_max else 0
            t_hat = torch.as_tensor(t_cur + gamma * t_cur)
            x_hat = x_cur + (t_hat ** 2 - t_cur ** 2).sqrt() * S_noise * torch.randn_like(x_cur)

            # Euler step.
            denoised = self.diffusion_net(x_hat, t_hat, class_labels).to(torch.float64)
            d_cur = (x_hat - denoised) / t_hat
            x_next = x_hat + (t_next - t_hat) * d_cur

            # Apply 2nd order correction.
            if i < num_steps - 1:
                denoised = self.diffusion_net(x_next, t_next, class_labels).to(torch.float64)
                d_prime = (x_next - denoised) / t_next
                x_next = x_hat + (t_next - t_hat) * (0.5 * d_cur + 0.5 * d_prime)

            x_next = x_next * self.sample_std
            t = torch.tensor(self.t, dtype=torch.float32).to(device)
            self.model.eval()
            x_next = self.model.decode(x_next, t, None)[0]
        return x_next