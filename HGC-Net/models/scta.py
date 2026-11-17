import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class SCTA(nn.Module):
    def __init__(self, dim, num_heads=1, kernel_size=3,
                 normalize_qk=True, bias=False):
        super().__init__()

        if dim % num_heads != 0:
            raise ValueError(
                f"dim={dim} must be divisible by num_heads={num_heads}"
            )

        if kernel_size % 2 != 1:
            raise ValueError("kernel_size must be odd")

        self.dim = dim
        self.num_heads = num_heads
        self.normalize_qk = normalize_qk

        self.temperature = nn.Parameter(
            torch.ones(num_heads, 1, 1)
        )

        self.q_proj = nn.Conv2d(
            dim, dim, kernel_size=1, bias=bias
        )
        self.k_proj = nn.Conv2d(
            dim, dim, kernel_size=1, bias=bias
        )
        self.v_proj = nn.Conv2d(
            dim, dim, kernel_size=1, bias=bias
        )

        self.q_dw = nn.Conv2d(
            dim,
            dim,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=dim,
            bias=bias
        )

        self.spatial_gate = nn.Sequential(
            nn.Conv2d(
                dim, dim, kernel_size=3,
                padding=1, groups=dim, bias=bias
            ),
            nn.Conv2d(dim, dim, kernel_size=1, bias=bias),
            nn.Sigmoid()
        )

        self.project_out = nn.Conv2d(
            dim, dim, kernel_size=1, bias=bias
        )

    def forward(self, x):
        b, c, h, w = x.shape

        q = self.q_dw(self.q_proj(x))
        q = q * self.spatial_gate(x)

        k = self.k_proj(x)
        v = self.v_proj(x)

        q = rearrange(
            q, "b (head ch) h w -> b head ch (h w)",
            head=self.num_heads
        )
        k = rearrange(
            k, "b (head ch) h w -> b head ch (h w)",
            head=self.num_heads
        )
        v = rearrange(
            v, "b (head ch) h w -> b head ch (h w)",
            head=self.num_heads
        )

        if self.normalize_qk:
            q = F.normalize(q, dim=-1)
            k = F.normalize(k, dim=-1)

        attn = torch.matmul(q, k.transpose(-2, -1))
        attn = attn * self.temperature
        attn = attn.softmax(dim=-1)

        out = torch.matmul(attn, v)

        out = rearrange(
            out,
            "b head ch (h w) -> b (head ch) h w",
            head=self.num_heads,
            h=h,
            w=w
        )

        return self.project_out(out)
