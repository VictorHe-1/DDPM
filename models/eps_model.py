import torch
from torch import nn
import math


class Swish(nn.Module):
    # Swish Activation function
    def forward(self, x):
        return x * torch.sigmoid(x)


class TimeEmbedding(nn.Module):
    '''
    Embedding for timestep $t$
    '''

    def __init__(self, n_channels):
        super().__init__()
        self.n_channels = n_channels
        self.linear1 = nn.Linear(self.n_channels // 4, self.n_channels)
        self.act = Swish()
        self.linear2 = nn.Linear(self.n_channels, self.n_channels)

    def forward(self, t):
        '''
        create sinusoidal position embeddings
        :param t:
        :return: torch.Tensor
        '''
        half_dim = self.n_channels // 8
        emb = math.log(10_000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device) * -emb)
        emb = t[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=1)

        # Transform with the MLP
        emb = self.act(self.linear1(emb))
        emb = self.linear2(emb)

        return emb


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_channels,
                 n_groups=32, dropout=0.1):
        super().__init__()
        self.norm1 = nn.GroupNorm(n_groups, in_channels)
        self.act1 = Swish()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3),
                               padding=(1, 1))  # won't change the image shape

        self.norm2 = nn.GroupNorm(n_groups, out_channels)
        self.act2 = Swish()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=(3, 3), padding=(1, 1))

        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1))
        else:
            self.shortcut = nn.Identity()

        self.time_emb = nn.Linear(time_channels, out_channels)
        self.time_act = Swish()

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, t):
        '''
        :param x: torch.Tensor [batch_size, in_channels, height, width]
        :param t: torch.Tensor [batch_size, time_channels]
        :return:
        '''
        h = self.conv1(self.act1(self.norm1(x)))

        h += self.time_emb(self.time_act(t))[:, :, None, None]
        h = self.conv2(self.dropout(self.act2(self.norm2(h))))

        return h + self.shortcut(x)


class AttentionBlock(nn.Module):
    def __init__(self, n_channels, n_heads=1, d_k=None):
        super().__init__()
        if d_k is None:
            d_k = n_channels
        self.projection = nn.Linear(n_channels, n_heads * d_k * 3)
        self.output = nn.Linear(n_heads * d_k, n_channels)

        self.scale = d_k ** -0.5

        self.n_heads = n_heads
        self.d_k = d_k

    def forward(self, x, t=None):
        '''
        :param x: has shape [batch_size, in_channels, height, width]
        :param t: has shape [batch_size, time_channels]
        :return:
        # t is not used but it's kept in the arguments because for the attention layer function signature
        # to match with `ResidualBlock`
        '''
        _ = t
        batch_size, n_channels, height, width = x.shape

        x = x.view(batch_size, n_channels, -1).permute(0, 2, 1)  # [batch_size, seq_length, n_channels]
        qkv = self.projection(x).view(batch_size, -1, self.n_heads,
                                      3 * self.d_k)  # [batch_size, seq_length, n_heads, dk * 3]
        q, k, v = torch.chunk(qkv, 3, dim=3)  # [batch_size, seq_length, n_head, dk * 3]
        attn = torch.einsum('bihd,bjhd->bijh', [q, k]) * self.scale  # [batch_size, seq_length, seq_length, n_head]
        attn = torch.softmax(attn, dim=2)

        v = torch.einsum('bijh,bjhd->bihd', [attn, v])  # [batch_size, seq_length, n_head, d_k]
        v = v.view(batch_size, -1, self.d_k * self.n_heads)  # [batch_size, -1, d_k * n_heads]
        res = self.output(v) + x  # [batch_size, seq_len, n_channels]
        res = res.permute(0, 2, 1).view(batch_size, n_channels, height, width)

        return res


class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_channels, has_attn):
        super().__init__()
        self.res = ResidualBlock(in_channels, out_channels, time_channels)
        if has_attn:
            self.attn = AttentionBlock(out_channels)
        else:
            self.attn = nn.Identity()

    def forward(self, x, t):
        x = self.res(x, t)
        x = self.attn(x)
        return x


class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_channels, has_attn):
        super().__init__()
        self.res = ResidualBlock(in_channels + out_channels, out_channels, time_channels)
        if has_attn:
            self.attn = AttentionBlock(out_channels)
        else:
            self.attn = nn.Identity()

    def forward(self, x, t):
        x = self.res(x, t)
        x = self.attn(x)
        return x


class MiddleBlock(nn.Module):
    def __init__(self, n_channels, time_channels):
        super().__init__()
        self.res1 = ResidualBlock(n_channels, n_channels, time_channels)
        self.attn = AttentionBlock(n_channels)
        self.res2 = ResidualBlock(n_channels, n_channels, time_channels)

    def forward(self, x, t):
        x = self.res1(x, t)
        x = self.attn(x)
        x = self.res2(x, t)
        return x


class UpSample(nn.Module):
    '''
    Scale up the feature map by 2 times
    '''

    def __init__(self, n_channels):
        super().__init__()
        self.conv = nn.ConvTranspose2d(n_channels, n_channels, (4, 4), (2, 2), (1, 1))

    def forward(self, x, t):
        _ = t
        return self.conv(x)


class DownSample(nn.Module):
    def __init__(self, n_channels):
        super().__init__()
        self.conv = nn.Conv2d(n_channels, n_channels, (3, 3), (2, 2), (1, 1))

    def forward(self, x, t):
        _ = t
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self, image_channels=3, n_channels=64,
                 ch_mults=(1, 2, 2, 4), is_attn=(False, False, True, True),
                 n_blocks=2):
        super().__init__()

        n_resolutions = len(ch_mults)

        self.image_proj = nn.Conv2d(image_channels, n_channels, kernel_size=(3, 3), padding=(1, 1))

        self.time_emb = TimeEmbedding(n_channels * 4)

        down = []
        out_channels = in_channels = n_channels
        for i in range(n_resolutions):
            out_channels = in_channels * ch_mults[i]
            for _ in range(n_blocks):
                down.append(DownBlock(in_channels, out_channels, n_channels * 4, is_attn[i]))
                in_channels = out_channels
            if i < n_resolutions - 1:
                down.append(DownSample(in_channels))

        self.down = nn.ModuleList(down)

        self.middle = MiddleBlock(out_channels, n_channels * 4)

        up = []
        in_channels = out_channels
        for i in reversed(range(n_resolutions)):
            out_channels = in_channels
            for _ in range(n_blocks):
                up.append(UpBlock(in_channels, out_channels, n_channels * 4, is_attn[i]))

            out_channels = in_channels // ch_mults[i]
            up.append(UpBlock(in_channels, out_channels, n_channels * 4, is_attn[i]))
            in_channels = out_channels
            if i > 0:
                up.append(UpSample(in_channels))

        self.up = nn.ModuleList(up)

        self.norm = nn.GroupNorm(8, n_channels)
        self.act = Swish()
        self.final = nn.Conv2d(in_channels, image_channels, kernel_size=(3, 3), padding=(1, 1))

    def forward(self, x, t):
        '''

        :param x: [batch_size, in_channels, height, width]
        :param t: [batch_size]
        '''
        t = self.time_emb(t)

        x = self.image_proj(x)

        h = [x]
        for m in self.down:
            x = m(x, t)
            h.append(x)

        x = self.middle(x, t)

        for m in self.up:
            if isinstance(m, UpSample):
                x = m(x, t)
            else:
                s = h.pop()
                x = torch.cat((x, s), dim=1)
                x = m(x, t)

        return self.final(self.act(self.norm(x)))
