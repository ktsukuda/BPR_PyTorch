import torch
from torch import nn


class BPR(nn.Module):

    def __init__(self, n_user, n_item, latent_dim):
        super().__init__()

        self.user_emb = nn.Embedding(num_embeddings=n_user, embedding_dim=latent_dim)
        self.item_emb = nn.Embedding(num_embeddings=n_item, embedding_dim=latent_dim)

    def forward(self, user_indices, item_indices):
        user_feature = self.user_emb(user_indices)
        item_feature = self.item_emb(item_indices)

        out = torch.sum(user_feature * item_feature, 1)
        return out
