import torch
from torch import nn


class BPR(nn.Module):

    def __init__(self, n_user, n_item, latent_dim):
        super().__init__()

        self.user_emb = nn.Embedding(num_embeddings=n_user, embedding_dim=latent_dim)
        self.item_emb = nn.Embedding(num_embeddings=n_item, embedding_dim=latent_dim)
