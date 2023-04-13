# smart-lyrics

Generative AI for song lyrics

## Setup

1. Clone the repo
2. Run these two commands

```
make install
. venv/bin/activate
```

3. Run the main file

```
make run
```

## Training

### Embedding Model

To train the embedding model you use `train.py`. Here is an example:

```
python3 src/train.py --model_type embedding_model --data_path ./data/lyricskaggle.csv --batch_size 32
```

The trained model will be saved in the `./models` directory in the `.pt` format. For instance, `embedding_model_2023-04-13_13-47-56.pt`.

The embedding model will also save the tensor of learned embeddings
in `./models/genre_embeddings.pt`. This is loaded in and used by the
VAE later.

## LSTM VAE

## LSTM VAE

Training the LSTM-VAE is similar. Here is an example:

```
python3 src/train.py --model_type lstm_vae --data_path ./data/lyricskaggle.csv --embedding_path ./models/genre_embeddings.pt --batch_size 32
```

Notice the additional argument `embedding_path` - the VAE needs this in the decoder step to tag songs with their genre embedding.

Like the embedding model, the VAE's state dict will be stored in
the models directory, like `lstm_vae_2023-04-13_13-47-56.pt`.

## Generating Songs

With both models trained, generating songs is simple. It can be
done like this:

```
import torch
from src.datasets import SongTokenizer
from src.models.lstm_vae import LSTM_VAE
from src.models.embedding_model import GenreEmbedding_LSTM

tokenizer = SongTokenizer()
kwargs, state = torch.load('./models/lstm_vae_2023-04-13_15-35-13.pt')
model = LSTM_VAE(**kwargs)
model.load_state_dict(state)
model.eval()
# here we give model.sample a random vector, but this should be a genre embedding vector
song = model.sample(torch.randn(1, 32))
for i in range(song.shape[0]):
    print(tokenizer.decode_text(song[i]))
```
