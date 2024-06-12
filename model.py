import torch
import torch.nn as nn
import torch.nn.functional as F

class DownSample(nn.Module):

    def __init__(self, in_channels, out_channels, embed_dim=128):
        super().__init__()
        self.block = nn.Sequential(
            nn.MaxPool2d(kernel_size=2),
            ConvBlock(in_channels, in_channels, residual=True),
            ConvBlock(in_channels, out_channels),
        )

        self.embedded_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(embed_dim, out_channels),
        )

    def forward(self, x, t):
        """
        args: x shape (B, C, H, W)
        args: t shape (B, C)
        return
        """
        x = self.block(x)
        embedded = self.embedded_layer(t)
        # reshape the embedded to (B, C, 1, 1) and enlarge the dimension of x to (B, C, H, W)
        embedded = embedded.view(-1, embedded.size(1), 1, 1).repeat(1, 1, x.size(2), x.size(3))
        return x + embedded


class UpSample(nn.Module):
    def __init__(self, in_channels, out_channels, embed_dim=128):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.block = nn.Sequential(
            ConvBlock(in_channels, in_channels, residual=True),
            ConvBlock(in_channels, out_channels, in_channels//2),
        )

        self.embedded_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(embed_dim, out_channels),
        )

    def forward(self, x, skip_x, t):
        """
        args: x shape (B, C, H, W)
        args: t shape (B, C)
        return (B, C, H, W)
        """
        x = self.up(x)
        x = torch.concat([x, skip_x], dim=1)
        x = self.block(x)

        embedded = self.embedded_layer(t)
        embedded = embedded.view(-1, embedded.size(1), 1, 1).repeat(1, 1, x.size(2), x.size(3))
        return x + embedded


class SelfAttention(nn.Module):
    def __init__(self, channels, size):
        super().__init__()
        self.size = size
        self.channels = channels

        self.MhA = nn.MultiheadAttention(channels, num_heads=4, batch_first=True)
        self.layer_norm = nn.LayerNorm(channels)
        self.block = nn.Sequential(
            nn.LayerNorm(channels),
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Linear(channels, channels),
        )

    def forward(self, x):
        """
        args: x shape (B, C, H, W)
        return (B, C, H, W)
        assume H = W
        """
        x = x.view(-1, self.channels, self.size*self.size).swapaxes(1, 2)
        x = self.layer_norm(x)
        attention_x, _ = self.MhA(x, x, x)
        attention_x = attention_x + x
        x = x + self.block(attention_x)

        return x.swapaxes(1, 2).view(-1, self.channels, self.size, self.size)


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None, residual=False):
        super().__init__()
        self.residual = residual
        if mid_channels is None:
            mid_channels = out_channels
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.GroupNorm(1, mid_channels),
            nn.GELU(),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(1, out_channels),
        )

    def forward(self, x):
        if self.residual:
            return F.gelu(x + self.block(x))
        else:
            return self.block(x)


class UNet(nn.Module):
    def __init__(self, c_in=1, c_out=1, time_dim=128, device="cuda"):
        super().__init__()
        self.device = device
        self.time_dim = time_dim
        self.inc = ConvBlock(c_in, 32)
        self.down1 = DownSample(32, 64)
        self.sa1 = SelfAttention(64, 14)
        self.down2 = DownSample(64, 128)
        self.sa2 = SelfAttention(128, 7)


        self.bot1 = ConvBlock(128, 64)

        self.up1 = UpSample(128, 32)
        self.sa3 = SelfAttention(32, 14)
        self.up2 = UpSample(64, 16)
        self.sa4 = SelfAttention(16, 28)

        self.outc = nn.Conv2d(16, c_out, kernel_size=1)

    def pos_encoding(self, t, channels):
        inv_freq = 1.0 / (
            10000
            ** (torch.arange(0, channels, 2, device=self.device).float() / channels)
        )
        pos_enc_a = torch.sin(t.repeat(1, channels // 2) * inv_freq)
        pos_enc_b = torch.cos(t.repeat(1, channels // 2) * inv_freq)
        pos_enc = torch.cat([pos_enc_a, pos_enc_b], dim=-1)
        return pos_enc

    def forward(self, x, t):
        t = t.unsqueeze(-1).type(torch.float)
        t = self.pos_encoding(t, self.time_dim)

        x1 = self.inc(x)  # torch.Size([1, 32, 28, 28])
        # down1 and sa1
        x2 = self.down1(x1, t)
        x2 = self.sa1(x2)  # torch.Size([1, 64, 14, 14])
        # down2 and sa2
        x3 = self.down2(x2, t)
        x3 = self.sa2(x3)  # torch.Size([1, 128, 7, 7])

        # bottleneck
        x4 = self.bot1(x3)  # torch.Size([1, 64, 7, 7])


        # up1 and sa4
        x5 = self.up1(x4, x2, t)
        x5 = self.sa3(x5) # torch.Size([1, 32, 14, 14])

        # up2 and sa5
        x6 = self.up2(x5, x1, t)
        x6 = self.sa4(x6) # torch.Size([1, 16, 28, 28])
        # output
        output = self.outc(x6)
        return output

class UNet_Condition(nn.Module):
    def __init__(self, c_in=1, c_out=1, time_dim=128, numclass=10, device="cuda"):
        super().__init__()
        self.device = device
        self.time_dim = time_dim
        self.inc = ConvBlock(c_in, 32)
        self.down1 = DownSample(32, 64)
        self.sa1 = SelfAttention(64, 14)
        self.down2 = DownSample(64, 128)
        self.sa2 = SelfAttention(128, 7)


        self.bot1 = ConvBlock(128, 64)

        self.up1 = UpSample(128, 32)
        self.sa3 = SelfAttention(32, 14)
        self.up2 = UpSample(64, 16)
        self.sa4 = SelfAttention(16, 28)

        self.outc = nn.Conv2d(16, c_out, kernel_size=1)

        self.label_embedding = nn.Embedding(numclass, time_dim)

    def pos_encoding(self, t, channels):
        inv_freq = 1.0 / (
            10000
            ** (torch.arange(0, channels, 2, device=self.device).float() / channels)
        )
        pos_enc_a = torch.sin(t.repeat(1, channels // 2) * inv_freq)
        pos_enc_b = torch.cos(t.repeat(1, channels // 2) * inv_freq)
        pos_enc = torch.cat([pos_enc_a, pos_enc_b], dim=-1)
        return pos_enc

    def forward(self, x, t, y):
        """"
        args: x shape (B, C, H, W)
        args: t shape (B,)
        args: y shape (B,)
        """
        t = t.unsqueeze(-1).type(torch.float)
        t = self.pos_encoding(t, self.time_dim)

        # add label embedding
        if y is not None:
            label_embed = self.label_embedding(y)
            t = t + label_embed

        x1 = self.inc(x)  # torch.Size([1, 32, 28, 28])
        # down1 and sa1
        x2 = self.down1(x1, t)
        x2 = self.sa1(x2)  # torch.Size([1, 64, 14, 14])
        # down2 and sa2
        x3 = self.down2(x2, t)
        x3 = self.sa2(x3)  # torch.Size([1, 128, 7, 7])

        # bottleneck
        x4 = self.bot1(x3)  # torch.Size([1, 64, 7, 7])


        # up1 and sa4
        x5 = self.up1(x4, x2, t)
        x5 = self.sa3(x5) # torch.Size([1, 32, 14, 14])

        # up2 and sa5
        x6 = self.up2(x5, x1, t)
        x6 = self.sa4(x6) # torch.Size([1, 16, 28, 28])
        # output
        output = self.outc(x6)
        return output

if __name__ == '__main__':
    # fake fashion minist input
    x = torch.randn(64, 1, 28, 28).to('cuda')
    model = UNet_Condition().to('cuda')
    print(model)