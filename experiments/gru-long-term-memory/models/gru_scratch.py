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