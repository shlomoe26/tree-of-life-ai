import torch
import torch.nn as nn
import torch.nn.functional as F

class BigramLM(nn.Module):
    """
    B0: the sanity baseline (bigram).
    A trivial model that predicts the next character from the previous one only.
    Useful to check that the loss and logging pipeline works.
    """
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets=None):
        logits = self.token_embedding_table(idx)
        loss, intent_loss = None, torch.tensor(0.0, device=idx.device)

        if targets is not None:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss, intent_loss
