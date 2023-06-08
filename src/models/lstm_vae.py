import torch
import torch.nn as nn


class LSTM_VAE(nn.Module):
    # from paper: (artist) genre_embed_size: 50
    def __init__(self, vocab_size, seq_len, embed_size, hidden_size, latent_size, genre_embed_size):
        super(LSTM_VAE, self).__init__()
        self.kwargs = {
            'vocab_size': vocab_size,
            'seq_len': seq_len,
            'embed_size': embed_size,
            'hidden_size': hidden_size,
            'latent_size': latent_size,
            'genre_embed_size': genre_embed_size
        }
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.latent_size = latent_size
        self.genre_embed_size = genre_embed_size

        self.embedding = nn.Embedding(self.vocab_size, self.embed_size)

        # Encoder
        # from paper: bidirectional, 100 hidden units
        self.encoder = nn.LSTM(
            self.embed_size, self.hidden_size, batch_first=True)
        self.fc_mu = nn.Linear(self.hidden_size, self.latent_size)
        self.fc_logvar = nn.Linear(self.hidden_size, self.latent_size)

        # Decoder
        self.fc_z = nn.Linear(
            self.latent_size + self.genre_embed_size, self.hidden_size)
        self.decoder = nn.LSTM(self.embed_size, self.hidden_size, batch_first=True)
        # Might need to make self.hidden_size//2 bigger, not sure
        self.fc_out = nn.Linear(self.hidden_size, self.vocab_size)

    def encode(self, x):
        x = self.embedding(x)
        _, (h, _) = self.encoder(x)
        h = h.view(-1, self.hidden_size)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, x, z, genre):
        z = torch.cat((z, genre), dim=1)
        h_0 = self.fc_z(z)
        h_0 = h_0.view(1, -1, self.hidden_size)
        # z = z.repeat(1, self.seq_len, 1)
        tmp = torch.zeros_like(x)
        tmp[:, 1:] = x[:, 0:-1]
        # x[:, 0] = 0
        tmp = self.embedding(tmp)
        op, _ = self.decoder(tmp, (h_0,h_0))
        op = self.fc_out(op)
        return op

    def forward(self, x, genre):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        # print(f'For {genre} genre: latent space vector: {z}')
        op = self.decode(z, genre)
        return op, mu, logvar

    def sample(self, genre):
        self.eval()
        z = torch.randn(1, self.latent_size)
        # z_arr = [[-0.5407, 0.5804, 0.4549, -0.1384, 0.4288, -0.9226, -0.1957, 1.1399]]
        # z = torch.tensor(z_arr)
        # print(f'latent space vector: {z}')
        x = torch.zeros(1, 512, dtype=torch.long)
        for i in range(1, 512):
            op = self.decode(x[:, 0:i], z, genre)
            op = torch.softmax(op[:, -1, :], dim=1)
            op = torch.multinomial(op, 1)
            x[:, i] = op
        return x.detach().numpy()

    def kl_divergence(self, mu, logvar):
        return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    def reconstruction_loss(self, op, x):
        return nn.CrossEntropyLoss()(op, x)

    def vae_loss(self, op, x, mu, logvar):
        recon_loss = self.reconstruction_loss(op, x)  # are words the same
        # difference between 2 distributions, form of a regularization. prevents overfitting
        kl_loss = self.kl_divergence(mu, logvar)
        return recon_loss + kl_loss


def test(model, val_loader):
    model.eval()
    total_loss = 0
    count = 0
    # test the model on validation set and return accuracy/F1 score or something
    for x, genre_embedding in val_loader:
        output, mu, logvar = model.forward(x, genre_embedding)
        loss = model.vae_loss(output, x, mu, logvar)
        total_loss += loss
        # NOTSURE: get batch_size??
        count += genre_embedding.shape[0]
    print('{:>12s} {:>7.5f}'.format('Testing loss:', total_loss/count))
