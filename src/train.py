import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim
from models.lstm_vae import LSTM_VAE
from models.embedding_model import GenreEmbedding_LSTM
from datasets import Dataset, create_loaders
from datetime import datetime


def train_lstm_vae(model, train_loader, test_loader, epochs, lr):
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    beta = 0
    for epoch in range(epochs):
        total_loss = 0
        total_recon_loss = 0
        total_kl_loss = 0
        count = 0
        for x, genre_embedding in train_loader:
            optimizer.zero_grad()
            output, mu, logvar = model(x, genre_embedding)
            recon_loss, kl_loss = model.vae_loss(output.transpose(1, 2), x, mu, logvar)
            loss = recon_loss + beta*kl_loss
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            total_recon_loss += recon_loss.item()
            total_kl_loss += kl_loss.item()
            count += 1
        beta += (0.1/epochs)
        print(f'Epoch {epoch}: {total_loss/count} Recon: {total_recon_loss/count} KL: {total_kl_loss/count}')


def train_embedding_model(model, train_loader, val_loader, epochs, lr):
    print('Training embedding model...')
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    test_embedding_model(model, val_loader)
    for epoch in range(epochs):
        model.train()
        for batch in train_loader:
            optimizer.zero_grad()
            x, y = batch
            y_hat = model(x)
            loss = criterion(y_hat, y)
            loss.backward()
            optimizer.step()

        print(f'Epoch {epoch}: {loss.item()}')
        test_embedding_model(model, train_loader)
        test_embedding_model(model, val_loader)


def test_embedding_model(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in test_loader:
            x, y = batch
            y_hat = model(x)
            predicted = torch.argmax(y_hat, dim=1)
            total += y.size(0)
            correct += (predicted == y).sum().item()

    print(f'Accuracy: {100 * correct / total}')


def save_model(model, model_name):
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    torch.save([model.kwargs, model.state_dict()],
               f'./models/{model_name}_{timestamp}.pt')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', type=str,
                        default='lstm_vae', help='Model to train')
    parser.add_argument('--data_path', type=str,
                        default='./data/data.csv', help='Path to data')
    parser.add_argument('--embedding_path', type=str,
                        default='./models/genre_embeddings.pt', help='Path to embeddings (for VAE)')
    parser.add_argument('--epochs', type=int, default=10,
                        help='Number of epochs to train')
    parser.add_argument('--batch_size', type=int,
                        default=1, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate')
    args = parser.parse_args()

    GENRE_EMBEDDING_SIZE = 128
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device.type)

    if (args.model_type == 'lstm_vae'):
        # Load data
        dataset = Dataset(args.data_path, device, max_len=256)
        embeds = torch.load('./models/genre_embeddings.pt')
        new_y = torch.zeros(dataset.y.shape[0], embeds.shape[1])
        for i in range(dataset.y.shape[0]):
            new_y[i] = embeds[dataset.y[i]]
        new_y = new_y.to(device)
        train_loader, test_loader = create_loaders(
            dataset.x, new_y, args.batch_size)

        # Train and save model
        model = LSTM_VAE(dataset.vocab_size, dataset.max_len,
                         300, 512, 8, GENRE_EMBEDDING_SIZE, device)
        train_lstm_vae(model, train_loader, test_loader, args.epochs, args.lr)
        save_model(model, args.model_type)

    elif (args.model_type == 'embedding_model'):
        # Load data
        dataset = Dataset(args.data_path, device)
        train_loader, test_loader = create_loaders(
            dataset.x, dataset.y, args.batch_size)
        model = GenreEmbedding_LSTM(
            dataset.vocab_size, 300, 256, GENRE_EMBEDDING_SIZE, dataset.num_labels, device)

        # Train and save model
        train_embedding_model(model, train_loader,
                              test_loader, args.epochs, args.lr)
        torch.save(model.get_embeddings().detach().to('cpu'),
                   './models/genre_embeddings.pt')
        save_model(model, args.model_type)

    else:
        print('Invalid model type')


if __name__ == '__main__':
    main()
