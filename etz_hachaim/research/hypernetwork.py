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

    @torch.no_grad()
    def generated_std(self, n_worlds: int, n_sefirot: int = 10) -> float:
        """Standard deviation of all generated weights over every (world, sefirah) position."""
        ws = torch.stack([self.forward(w, s) for w in range(n_worlds) for s in range(n_sefirot)])
        return ws.std().item()

    @torch.no_grad()
    def calibrate_init(self, n_worlds: int, n_sefirot: int = 10) -> float:
        """Rescale the output layer so that the generated weights start at the scale a directly
        parameterised layer would have (std = base_dim ** -0.5, the init of the 'direct' ablation).
        This targets the same quantity as the hyperfan-in initialisation of Chang et al. (2020),
        but by empirical calibration rather than their analytic formula. Returns the scale applied."""
        target = self.base_dim ** -0.5
        scale = target / self.generated_std(n_worlds, n_sefirot)
        out = self.weight_generator[-1]
        out.weight.mul_(scale)
        out.bias.mul_(scale)
        return scale
