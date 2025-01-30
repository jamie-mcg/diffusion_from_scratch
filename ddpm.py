import torch
from tqdm import tqdm

from utils import *

NOISE_SCHEDULES = {
    "linear": lambda beta_init, beta_final, steps: torch.linspace(
        beta_init, beta_final, steps
    ),
    "cosine": cosine_schedule,
}


class DDPM:
    def __init__(
        self,
        img_size: int = 32,
        n_steps: int = 1000,
        beta_init: float = 1e-4,
        beta_final: float = 1e-2,
        schedule: str = "linear",
    ):
        self.n_steps = n_steps
        self.beta_init = beta_init
        self.beta_final = beta_final
        self.img_size = img_size

        self.betas = NOISE_SCHEDULES[schedule](beta_init, beta_final, n_steps)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, axis=0)

    def get_noised_image(self, x: torch.Tensor, t: torch.Tensor):
        epsilon = torch.randn_like(x)
        # x_t = sqrt(alpha_t) * x_0 + sqrt(1 - alpha_t) * epsilon with
        # broadcasting up to the shape of x_0 - when t is a vector of
        # timesteps for each sample in the batch
        return (
            torch.sqrt(self.alphas_cumprod[t])[:, None, None, None] * x
            + torch.sqrt(1 - self.alphas_cumprod[t])[:, None, None, None] * epsilon
        )

    def sample_timesteps(self, n_samples: int):
        return torch.randint(1, self.n_steps, (n_samples,))

    def sample(self, model: torch.nn.Module, n: int):
        model.eval()
        with torch.no_grad():
            x = torch.randn(n, 3, self.img_size, self.img_size).to(model.device)
            for i in tqdm(reversed(range(1, self.n_steps))):
                t = torch.full((n,), i, dtype=torch.int64).to(model.device)
                pred_noise = model(x, t)
                if i > 1:
                    z = torch.randn_like(x)
                else:
                    z = torch.zeros_like(x)
                x = (1 / torch.sqrt(self.alphas[t][:, None, None, None])) * (
                    x
                    - (
                        (1 - self.alphas[t][:, None, None, None])
                        / torch.sqrt(1 - self.alphas_cumprod[t][:, None, None, None])
                    )
                    * pred_noise
                ) + torch.sqrt(self.betas[t][:, None, None, None]) * z
        model.train()
        x = (x.clamp(-1, 1) + 1) / 2
        x = (x * 255).type(torch.uint8)
        return x

    def forward(self, x):
        pass

    def backward(self, x):
        pass
