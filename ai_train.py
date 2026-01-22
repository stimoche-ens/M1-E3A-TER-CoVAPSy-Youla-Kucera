#!/usr/bin/env python3
class TrajectorySeq2Seq(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=512, output_dim=360, num_layers=2):
        super().__init__()
        # LSTM processes the sequence of 50 steps
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=0.1)
        
        # The Linear layer maps the hidden state at EACH timestep to the 360 measures
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # x shape: (Batch, 50, 2)
        lstm_out, (hn, cn) = self.lstm(x)
        
        # lstm_out shape: (Batch, 50, hidden_dim)
        # We pass every timestep through the fully connected layer
        predictions = self.fc(lstm_out) 
        
        # predictions shape: (Batch, 50, 360)
        return predictions
