import torch
import torch.nn as nn

class GRUCellScratch(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()

        # Reset gate
        self.W_r = nn.Linear(input_size, hidden_size)
        self.U_r = nn.Linear(hidden_size, hidden_size, bias=False)

        # Update gate
        self.W_z = nn.Linear(input_size, hidden_size)
        self.U_z = nn.Linear(hidden_size, hidden_size, bias=False)

        # Candidate hidden state
        self.W_h = nn.Linear(input_size, hidden_size)
        self.U_h = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, x_t, h_prev):
        # Reset gate
        r_t = torch.sigmoid(
            self.W_r(x_t) + self.U_r(h_prev)
        )

        # Update gate
        z_t = torch.sigmoid(
            self.W_z(x_t) + self.U_z(h_prev)
        )

        # Candidate hidden state
        h_candidate = torch.tanh(
            self.W_h(x_t)
            + self.U_h(r_t * h_prev)
        )

        # New hidden state
        h_t = (
            z_t * h_prev
            + (1 - z_t) * h_candidate
        )

        return h_t


class GRUScratch(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()

        self.hidden_size = hidden_size
        self.cell = GRUCellScratch(
            input_size,
            hidden_size
        )

    def forward(self, x):
        batch_size = x.size(0)

        h = torch.zeros(
            batch_size,
            self.hidden_size,
            device=x.device
        )

        reset_gates = []
        update_gates = []

        for t in range(x.size(1)):
            h, r_t, z_t, _ = self.cell(
                x[:, t, :],
                h
            )

            reset_gates.append(r_t)
            update_gates.append(z_t)

        reset_gates = torch.stack(reset_gates, dim=1)
        update_gates = torch.stack(update_gates, dim=1)

        return h, reset_gates, update_gates

class RNNClassifier(nn.Module):
    def __init__(self, input_size=1, hidden_size=32):
        super().__init__()

        self.rnn = nn.RNN(
            input_size=input_size,
            hidden_size=hidden_size,
            batch_first=True
        )

        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        output, h_n = self.rnn(x)

        # Last hidden state
        h_last = h_n[-1]

        logits = self.fc(h_last).squeeze(1)

        return logits