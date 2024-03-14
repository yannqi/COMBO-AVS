import torch
import torch.nn as nn


class audio_mlp(nn.Module):
    def __init__(self, in_dim=128, middle_dim=4096, out_dim=256):
        """MLP for audio transformation"""
        super(audio_mlp, self).__init__()
        self.embeddings = nn.Sequential(
            nn.Linear(in_dim, middle_dim), nn.ReLU(True), nn.Linear(middle_dim, middle_dim), nn.ReLU(True), nn.Linear(middle_dim, out_dim)
        )

    def forward(self, x):
        return self.embeddings(x)
