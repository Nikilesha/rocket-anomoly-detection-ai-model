import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

class LSTMAnomalyDetector(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_layers=2):
        super(LSTMAnomalyDetector, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.encoder = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.decoder = nn.LSTM(hidden_dim, hidden_dim, num_layers, batch_first=True)
        self.reconstruction = nn.Linear(hidden_dim, input_dim)

    def forward(self, x):
        encoded_seq, (h, c) = self.encoder(x)
        decoded_seq, _ = self.decoder(encoded_seq, (h, c))
        reconstructed = self.reconstruction(decoded_seq)
        return reconstructed


def train_lstm_autoencoder(train_data, input_dim, num_epochs=50, batch_size=32, lr=1e-3, model_path="lstm_model.pth"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_tensor = torch.tensor(train_data, dtype=torch.float32)
    train_loader = DataLoader(TensorDataset(train_tensor), batch_size=batch_size, shuffle=True)
    model = LSTMAnomalyDetector(input_dim=input_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        for (batch,) in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            output = model(batch)
            loss = criterion(output, batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        print(f"Epoch [{epoch+1}/{num_epochs}] - Loss: {epoch_loss/len(train_loader):.6f}")
    torch.save(model.state_dict(), model_path)
    print(f"✅ LSTM model saved to {model_path}")
    return model


def detect_anomalies(model, data, threshold=0.02):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    data_tensor = torch.tensor(data, dtype=torch.float32).to(device)
    with torch.no_grad():
        output = model(data_tensor)
        loss = torch.mean((output - data_tensor) ** 2, dim=(1, 2))
    return (loss > threshold).cpu().numpy(), loss.cpu().numpy()
