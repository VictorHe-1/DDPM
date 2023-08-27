from typing import List

import torch

from models.eps_model import UNet
from models.diffusion import DenoiseDiffusion


def get_device():
    device = torch.device("cuda" if torch.cuda.is_available()
                          else "mps" if torch.backends.mps.is_available()
                          else "cpu")
    print(f"current device is {device}")
    return device


class Configs:
    device: torch.device = get_device()

    eps_model: UNet
    diffusion: DenoiseDiffusion

    image_channels: int = 1
    image_size: int = 32
    n_channels: int = 64
    channel_multipliers: List[int] = [1, 2, 2, 4]
    is_attention: List[int] = [False, False, False, True]

    n_steps: int = 1_000
    batch_size: int = 64
    n_samples: int = 16  # number of samples to generate
    learning_rate: float = 2e-5

    epochs: int = 1_000

    data_loader: torch.utils.data.DataLoader

    optimizer: torch.optim.Adam

    def update(self, config_item: dict[str, any]):
        for key, value in config_item.items():
            setattr(self, key, value)

    def init(self):
        self.eps_model = UNet(
            image_channels=self.image_channels,
            n_channels=self.n_channels,
            ch_mults=self.channel_multipliers,
            is_attn=self.is_attention,
        ).to(self.device)

        self.diffusion = DenoiseDiffusion(
            eps_model=self.eps_model,
            n_steps=self.n_steps,
            device=self.device
        )

        self.data_loader = torch.utils.data.DataLoader(self.dataset, self.batch_size,
                                                       shuffle=True, pin_memory=True)
        self.optimizer = torch.optim.Adam(self.eps_model.parameters(), lr=self.learning_rate)

    def sample(self):
        with torch.no_grad():
            x = torch.randn([self.n_samples, self.image_channels, self.image_size,
                             self.image_size], device=self.device)

            for t_ in range(self.n_steps):
                t = self.n_steps - t_ - 1
                x = self.diffusion.p_sample(x, x.new_full((self.n_samples,), t, dtype=torch.long))

    def train(self, epoch):
        for i, data in enumerate(self.data_loader):
            data = data.to(self.device)
            self.optimizer.zero_grad()
            loss = self.diffusion.loss(data)
            loss.backward()
            self.optimizer.step()
            print('Train Epoch: {} [Step: {}/{}] \tLoss: {:.6f}'.format(epoch,
                                                                        i,
                                                                        len(self.data_loader),
                                                                        loss.item()))

    def run(self):
        for e in range(self.epochs):
            self.train(e)
            self.sample()
            self.save_checkpoint(f"diffusion_epoch{e+1}.pt")

    def save_checkpoint(self, model_name):
        torch.save(self.diffusion.state_dict(), model_name)

    def load_checkpoint(self, model_path):
        self.diffusion.load_state_dict(torch.load(model_path))
