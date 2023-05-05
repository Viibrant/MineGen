from torch.utils.data import DataLoader, Subset
from torch import nn
from tqdm import tqdm


class VanillaAutoencoder(nn.Module):
    def __init__(self, embedding_dim):
        super(VanillaAutoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv3d(1, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv3d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv3d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv3d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv3d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv3d(256, embedding_dim, kernel_size=3, stride=2, padding=1),
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose3d(embedding_dim, 256, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose3d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose3d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose3d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose3d(32, 16, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose3d(16, 1, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat, z


def train_voxel_embedding(embedding_dim, num_epochs, batch_size, learning_rate, device):
    # dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
    dataloader = DataLoader(
        Subset(dataset, range(20)), batch_size=batch_size, shuffle=True, pin_memory=True
    )

    model = VanillaAutoencoder(embedding_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()

    for epoch in range(num_epochs):
        epoch_loss = 0
        for batch in tqdm(sub):
            batch = batch.unsqueeze(1).float().to(device) / 255.0
            optimizer.zero_grad()
            x_hat, z = model(batch)
            loss = criterion(x_hat, batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * batch.size(0)

        print(f"Epoch {epoch}: Loss={epoch_loss}")

    return model
