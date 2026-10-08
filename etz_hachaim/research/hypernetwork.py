import torch
import torch.nn as nn

class FractalHyperNet(nn.Module):
    """
    Weight generator (hypernetwork).
    Takes the position in the Tree (World, Sefirah) and generates the weight matrix.
    This forces massive parameter sharing. The ablation in listops_experiment/ shows that
    this component is what costs the architecture most of its accuracy.
    """
    def __init__(self, base_dim=64, n_worlds=4, n_sefirot=10):
        super().__init__()
        self.base_dim = base_dim

        # Embeddings that encode the position in the Tree of Life
        self.world_emb = nn.Embedding(n_worlds, 32)
        self.sefirah_emb = nn.Embedding(n_sefirot, 32)

        # The generator (an MLP) that outputs the weights
        self.weight_generator = nn.Sequential(
            nn.Linear(64, 256),
            nn.GELU(),
            nn.Linear(256, base_dim * base_dim)
        )

    def forward(self, world_idx: int, sefirah_idx: int) -> torch.Tensor:
        """
        Generate a (base_dim x base_dim) weight matrix specific to
        Sefirah 'sefirah_idx' in World 'world_idx'.
        """
        w = torch.tensor(world_idx, dtype=torch.long, device=self.world_emb.weight.device)
        s = torch.tensor(sefirah_idx, dtype=torch.long, device=self.sefirah_emb.weight.device)

        emb = torch.cat([self.world_emb(w), self.sefirah_emb(s)], dim=-1)
        weights_flat = self.weight_generator(emb)

        return weights_flat.view(self.base_dim, self.base_dim)
