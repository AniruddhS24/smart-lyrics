import torch
from src.datasets import SongTokenizer
from src.models.lstm_vae import LSTM_VAE
from src.models.embedding_model import GenreEmbedding_LSTM

tokenizer = SongTokenizer()
kwargs, state = torch.load('./models/lstm_vae_2023-04-19_21-16-28.pt')
model = LSTM_VAE(**kwargs)
model.load_state_dict(state)
model.eval()
genre_embeddings = torch.load('./models/genre_embeddings.pt').to('cpu')

for s in range(10):
    print('Label:', s)
    song = model.sample(genre_embeddings[s].unsqueeze(0))  # pop female song
    for i in range(song.shape[0]):
        # print(song[i])
        print(tokenizer.decode_text(song[i]))
