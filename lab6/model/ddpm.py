# DDPM
from tqdm import tqdm

import torch
import torch.nn as nn


class DDPM(nn.Module):
    def __init__(self, unet_model, betas, noise_steps, device):
        super(DDPM, self).__init__()

        self.n_T = noise_steps
        self.device = device
        self.unet_model = unet_model

        # register_buffer allows accessing dictionary produced by ddpm_schedules
        # e.g. can access self.sqrtab later
        for k, v in ddpm_schedules(betas[0], betas[1], noise_steps).items():
            self.register_buffer(k, v)

        # loss function
        self.mse_loss = nn.MSELoss()

    def forward(self, x, cond):
        """training ddpm, sample time and noise randomly (return loss)"""
        # t ~ Uniform(0, n_T)
        timestep = torch.randint(1, self.n_T+1, (x.shape[0],)).to(self.device)  
        # eps ~ N(0, 1)
        noise = torch.randn_like(x)  

        x_t = (
            self.sqrtab[timestep, None, None, None] * x
            + self.sqrtmab[timestep, None, None, None] * noise
        ) 

        predict_noise = self.unet_model(x_t, cond, timestep/self.n_T)

        # return MSE loss between real added noise and predicted noise
        loss = self.mse_loss(noise, predict_noise)
        
        return loss

    def sample(self, cond, size, device):
        """sample initial noise and generate images based on conditions"""
        n_sample = len(cond)
        # x_T ~ N(0, 1), sample initial noise
        x_i = torch.randn(n_sample, *size).to(device)  
        for idx in tqdm(range(self.n_T, 0, -1), leave=False):
            timestep = torch.tensor([idx / self.n_T]).to(device)
            z = torch.randn(n_sample, *size).to(device) if idx > 1 else 0
            eps = self.unet_model(x_i, cond, timestep)
            x_i = (
                self.oneover_sqrta[idx] * (x_i - eps * self.mab_over_sqrtmab[idx])
                + self.sqrt_beta_t[idx] * z
            )
        return x_i



"""return pre-computed schedules for DDPM sampling in training process"""
def ddpm_schedules(beta1, beta2, T, schedule_type="linear"):

    assert (beta1 < beta2 < 1.0, "beta1 and beta2 must be in (0, 1)")

    if schedule_type == "linear":
        beta_t = (beta2 - beta1) * torch.arange(0, T + 1, dtype=torch.float32) / T + beta1
        sqrt_beta_t = torch.sqrt(beta_t)
        alpha_t = 1 - beta_t
        log_alpha_t = torch.log(alpha_t)
        alphabar_t = torch.cumsum(log_alpha_t, dim=0).exp()

        sqrtab = torch.sqrt(alphabar_t)
        oneover_sqrta = 1 / torch.sqrt(alpha_t)

        sqrtmab = torch.sqrt(1 - alphabar_t)
        mab_over_sqrtmab_inv = (1 - alpha_t) / sqrtmab

        return {
            "alpha_t": alpha_t,  # alpha_t
            "oneover_sqrta": oneover_sqrta,  # 1/sqrt{alpha_t}
            "sqrt_beta_t": sqrt_beta_t,  # sqrt{beta_t}
            "alphabar_t": alphabar_t,  # bar{alpha_t}
            "sqrtab": sqrtab,  # sqrt{bar{alpha_t}}
            "sqrtmab": sqrtmab,  # sqrt{1-bar{alpha_t}}
            "mab_over_sqrtmab": mab_over_sqrtmab_inv,  # (1-alpha_t)/sqrt{1-bar{alpha_t}}
        }
