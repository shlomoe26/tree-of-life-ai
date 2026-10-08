import torch
import torch.nn as nn
from etz_hachaim.research.fractal_worlds import EtzChaimFractalAI
from models.transformer_lm import Block

class EtzUnifiedLM(nn.Module):
    """
    Unified Etz HaChaim language model.
    E0, E4 and E5 are obtained by changing the switches (num_worlds, tzimtzum_active,
    temporal_attention).

    Note the `repeat(1, 10, 1)` in forward(): the same vector is copied onto the 10 Sefirot,
    so the graph never receives any structure from the data. This is the flaw that the
    ListOps experiment (listops_experiment/) was designed to remove.
    """
    def __init__(self, vocab_size: int, d_model: int = 64, num_worlds: int = 4, tzimtzum_active: bool = True, temporal_attention: bool = False):
        super().__init__()
        self.d_model = d_model
        self.temporal_attention = temporal_attention
        
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(256, d_model)
        
        if self.temporal_attention:
            # Causal self-attention block applied before the first World (variant E5)
            self.temporal_block = Block(d_model=d_model, num_heads=4, block_size=256)
        
        self.fractal_core = EtzChaimFractalAI(d_model=d_model, num_worlds=num_worlds, tzimtzum_active=tzimtzum_active)
        
        self.lm_head = nn.Linear(d_model, vocab_size)

    def forward(self, idx: torch.Tensor, targets=None):
        B, T = idx.shape
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        
        x = self.token_embedding(idx) + self.position_embedding(pos)
        
        if self.temporal_attention:
            x = self.temporal_block(x)
            
        x_flat = x.view(B * T, self.d_model)
        x_nodes = x_flat.unsqueeze(1).repeat(1, 10, 1)
        intent_signal = x_nodes.clone()
        
        x_out, intent_loss = self.fractal_core(x_nodes, intent_signal)
        
        x_malchut = x_out[:, 9, :] 
        x_malchut = x_malchut.view(B, T, self.d_model)
        logits = self.lm_head(x_malchut)
        
        loss = None
        if targets is not None:
            B, T, C = logits.shape
            loss = torch.nn.functional.cross_entropy(logits.view(B*T, C), targets.view(B*T))
            
        return logits, loss, intent_loss
