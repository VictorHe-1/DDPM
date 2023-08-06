'''
This is the code to generate images and create interpolations between given images
'''

import numpy as np
import torch
import matplotlib.pyplot as plt
from torchvision.transforms.functional import to_pil_image, resize
from utils.helper_functions import gather

from models.diffusion import DenoiseDiffusion

class Sampler:
    def __init__(self, diffusion, image_channels, image_size, device):
        self.device = device
        self.image_size = image_size
        self.image_channels = image_channels
        self.diffusion = diffusion

        self.n_steps = diffusion.n_steps
        self.eps_model = diffusion.eps_model

        self.beta = diffusion.beta
        self.alpha = diffusion.alpha
        self.alpha_bar = diffusion.alpha_bar
        alpha_bar_tm1 = torch.cat([self.alpha_bar.new_ones((1,)), self.alpha_bar[:-1]])

        self.beta_tilde = self.beta * (1 - alpha_bar_tm1) / (1 - self.alpha_bar)

        self.mu_tilde_coef1 = self.beta * (alpha_bar_tm1 ** 0.5) / ( 1 - self.alpha_bar)

        self.mu_tilde_coef2 = (self.alpha ** 0.5) * (1 - alpha_bar_tm1) / (1 - self.alpha_bar)

        self.sigma2 = self.beta

    def show_image(self, img, title=""):
        img = img.clip(0, 1)
        img = img.cpu().numpy()
        plt.imshow(img.transpose(1, 2, 0))
        plt.title(title)
        plt.show()

    def make_video(self, frames, path="video.mp4"):
        import imageio
        writer = imageio.get_writer(path, fps=len(frames) // 20)
        for f in frames:
            f = f.clip(0, 1)
            f = to_pil_image(resize(f, [368, 368]))
            writer.append_data(np.array(f))
        writer.close()

    def sample_animation(self, n_frames=1000, create_video=True):
        xt = torch.randn([1, self.image_channels, self.image_size, self.image_size], device=self.device)

        interval = self.n_steps // n_frames
        frames = []
        for t_inv in range(self.n_steps):
            t_ = self.n_steps - t_inv - 1

            t = xt.new_full((1, ), t_, dtype=torch.long)

            eps_theta = self.eps_model(xt, t)
            if t_ % interval == 0:
                x0 = self.p_x0(xt, t, eps_theta)
                frames.append(x0[0])
                if not create_video:
                    self.show_image(x0[0], f"{t_}")

            xt = self.p_sample(xt, t, eps_theta)

        if create_video:
            self.make_video(frames)

    def interpolate(self, x1: torch.Tensor, x2: torch.Tensor, lambda_: float, t_: int = 100):
        '''
        interpolate two images $x_0$ and $x'_0$
        '''
        n_samples = x1.shape[0]
        t = torch.full((n_samples, ), t_, device=self.device)
        xt = (1 - lambda_) * self.diffusion.q_sample(x1, t) + lambda_ * self.diffusion.q_sample(x2, t)

        return self._sample_x0(xt, t_)

    def interpolate_animate(self, x1: torch.Tensor, x2: torch.Tensor, n_frames: int = 100, t_: int = 100,
                            create_video: bool = True):

        self.show_image(x1, "x1")
        self.show_image(x2, "x2")
        x1 = x1[None, :, :, :]
        x2 = x2[None, :, :, :]
        t = torch.full((1, ), t_, device=self.device)
        x1t = self.diffusion.q_sample(x1, t)
        x2t = self.diffusion.q_sample(x2, t)

        frames = []
        for i in range(n_frames + 1):
            lambda_ = i / n_frames
            xt = (1 - lambda_) * x1t + lambda_ * x2t
            x0 = self._sample_x0(xt, t_)
            frames.append(x0[0])
            if not create_video:
                self.show_image(x0[0], f"{lambda_ :.2f}")

        if create_video:
            self.make_video(frames)

    def _sample_x0(self, xt, n_steps):
        '''
        gradually denoise
        '''
        n_samples = xt.shape[0]
        for t_ in range(n_steps):
            t = n_steps - t_ - 1
            xt = self.diffusion.p_sample(xt, xt.new_full((n_samples, ), t, dtype=torch.long))

        return xt

    def sample(self, n_samples: int = 16):
        xt = torch.randn([n_samples, self.image_channels, self.image_size, self.image_size], device=self.device)

        x0 = self._sample_x0(xt, self.n_steps)

        for i in range(n_samples):
            self.show_image(x0[i])

    def p_sample(self, xt, t, eps_theta):
        alpha_bar = gather(self.alpha_bar, t)

        alpha = gather(self.alpha, t)

        eps_coef = (1 - alpha) / (1 -  alpha_bar) ** .5

        mean = 1 / (alpha ** 0.5) * (xt - eps_coef * eps_theta)
        var = gather(self.sigma2, t)

        eps = torch.randn(xt.shape, device=xt.device)
        return mean + (var ** .5) * eps

    def p_x0(self, xt: torch.Tensor, t: torch.Tensor, eps: torch.Tensor):
        alpha_bar = gather(self.alpha_bar, t)

        return (xt - (1 - alpha_bar) ** .5 * eps) / (alpha_bar ** 0.5)


if __name__ == "__main__":
    self.eps_model = UNet(
        image_channels=self.image_channels,
        n_channels=self.n_channels,
        ch_mults=self.channel_multipliers,
        is_attn=self.is_attention,
    ).to(self.device)

    diffusion = DenoiseDiffusion()
    diffusion = torch.load("fff")
    sampler = Sampler(diffusion=diffusion,
                      image_channels=image_channels,
                      image_size=image_size,
                      device=device)
    with torch.no_grad():
        sampler.sample_animation()
        if False:
            data = next(iter(data_loader))
            sampler.interpolate_animate(data[0], data[1])





