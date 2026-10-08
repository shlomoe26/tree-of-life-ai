import torch
import torch.nn as nn
import torch.nn.functional as F

class TzimtzumLayer(nn.Module):
    """
    Tzimtzum (contraction): a bottleneck between two Worlds, with an auxiliary loss that
    checks that the original intent (Keter) can be reconstructed from the compressed state.
    """
    def __init__(self, d_high: int, d_low: int, intent_dim: int):
        super().__init__()
        # Compression (the contraction itself)
        self.compress = nn.Sequential(
            nn.Linear(d_high, d_high // 2),
            nn.GELU(),
            nn.Linear(d_high // 2, d_low)
        )

        # Readout head used to check that the intent is preserved
        self.intent_head = nn.Linear(d_low, intent_dim)

    def forward(self, x_high: torch.Tensor, original_intent: torch.Tensor):
        # 1. Contraction
        x_low = self.compress(x_high)

        # 2. Intent check
        # Can the original intent be recovered from the compressed information?
        reconstructed_intent = self.intent_head(x_low)
        intent_loss = F.mse_loss(reconstructed_intent, original_intent)

        return x_low, intent_loss
