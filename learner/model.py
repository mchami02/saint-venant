import torch
import torch.nn as nn

class SVEEncoder(nn.Module):
    def __init__(self, N_x, hidden_size, output_size, dropout_rate=0.1):
        super(SVEEncoder, self).__init__()
        self.N_x = N_x
        # Input size is 2*N_x since we concatenate two arrays of size N_x
        input_size = 2 * N_x
        
        # Deeper encoder with normalization and dropout
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.ln1 = nn.LayerNorm(hidden_size)
        self.dropout1 = nn.Dropout(dropout_rate)
        
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.ln2 = nn.LayerNorm(hidden_size)
        self.dropout2 = nn.Dropout(dropout_rate)
        
        self.fc3 = nn.Linear(hidden_size, hidden_size // 2)
        self.ln3 = nn.LayerNorm(hidden_size // 2)
        self.dropout3 = nn.Dropout(dropout_rate)
        
        self.fc4 = nn.Linear(hidden_size // 2, output_size)

    def forward(self, x):
        # x is a tuple of size 2, where each element is an array of size N_x
        if isinstance(x, tuple):
            # Concatenate the two arrays along the last dimension
            x = torch.cat(x, dim=-1)
        
        # Forward pass through deeper network
        x = self.fc1(x)
        x = self.ln1(x)
        x = torch.relu(x)
        x = self.dropout1(x)
        
        x = self.fc2(x)
        x = self.ln2(x)
        x = torch.relu(x)
        x = self.dropout2(x)
        
        x = self.fc3(x)
        x = self.ln3(x)
        x = torch.relu(x)
        x = self.dropout3(x)
        
        x = self.fc4(x)
        return x

class SVEPredictor(nn.Module):
    def __init__(self, input_size, hidden_size, N_x, dropout_rate=0.1):
        super(SVEPredictor, self).__init__()
        self.N_x = N_x
        
        # Deeper predictor with normalization and dropout
        self.fc1 = nn.Linear(input_size, hidden_size // 2)
        self.ln1 = nn.LayerNorm(hidden_size // 2)
        self.dropout1 = nn.Dropout(dropout_rate)
        
        self.fc2 = nn.Linear(hidden_size // 2, hidden_size)
        self.ln2 = nn.LayerNorm(hidden_size)
        self.dropout2 = nn.Dropout(dropout_rate)
        
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.ln3 = nn.LayerNorm(hidden_size)
        self.dropout3 = nn.Dropout(dropout_rate)
        
        # Output size is 2*N_x since we output two arrays of size N_x
        self.fc4 = nn.Linear(hidden_size, 2 * N_x)

    def forward(self, x):
        # Forward pass through deeper network
        x = self.fc1(x)
        x = self.ln1(x)
        x = torch.relu(x)
        x = self.dropout1(x)
        
        x = self.fc2(x)
        x = self.ln2(x)
        x = torch.relu(x)
        x = self.dropout2(x)
        
        x = self.fc3(x)
        x = self.ln3(x)
        x = torch.relu(x)
        x = self.dropout3(x)
        
        x = self.fc4(x)
        # Split the output into two arrays of size N_x and return as tuple
        h_pred, u_pred = torch.split(x, self.N_x, dim=-1)
        return (h_pred, u_pred)

class SVEModel(nn.Module):
    def __init__(self, N_x, hidden_size, latent_size, dropout_rate=0.1):
        super(SVEModel, self).__init__()
        self.N_x = N_x
        # Encoder takes tuple of 2 arrays of size N_x and outputs latent representation
        self.encoder = SVEEncoder(N_x, hidden_size, latent_size, dropout_rate=dropout_rate)
        # Predictor takes latent representation and outputs tuple of 2 arrays of size N_x
        self.predictor = SVEPredictor(latent_size, hidden_size, N_x, dropout_rate=dropout_rate)

    def forward(self, x):
        # x is a tuple of size 2, each array of size N_x
        x = self.encoder(x)
        x = self.predictor(x)
        # Returns a tuple of size 2, each array of size N_x
        return x