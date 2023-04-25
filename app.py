import torch
from src.datasets import SongTokenizer
from src.models.lstm_vae import LSTM_VAE
from src.models.embedding_model import GenreEmbedding_LSTM

tokenizer = SongTokenizer()
map_location=torch.device('cpu')
kwargs, state = torch.load('./models/lstm_vae_2023-04-19_21-16-28.pt', map_location=map_location)
model = LSTM_VAE(**kwargs)
model.load_state_dict(state)
model.eval()
genre_embeddings = torch.load('./models/genre_embeddings.pt')
song = model.sample(genre_embeddings[0].unsqueeze(0))  # pop female song
for i in range(song.shape[0]):
    print(tokenizer.decode_text(song[i]))
