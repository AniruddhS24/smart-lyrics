import torch
from src.datasets import SongTokenizer
from src.models.lstm_vae import LSTM_VAE
from src.models.embedding_model import GenreEmbedding_LSTM

tokenizer = SongTokenizer()
kwargs, state = torch.load('./models/lstm_vae_2023-04-13_15-35-13.pt')
model = LSTM_VAE(**kwargs)
model.load_state_dict(state)
model.eval()
# right now we give it a random vector, but this should be a genre embedding vector
song = model.sample(torch.randn(1, 32))
for i in range(song.shape[0]):
    print(tokenizer.decode_text(song[i]))
