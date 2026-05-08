# ==============================================================================
# File: model/attention.py
# Part 2: Recurrent Network-Based Architecture (LSTM + Additive Attention)
# Step 2: Additive Attention Mechanism
# ==============================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F

class AdditiveAttention(nn.Module):
    def __init__(self, hidden_size: int):
        """
        hidden_size (h): 512 as per the assignment parameters.
        """
        super(AdditiveAttention, self).__init__()
        
        # W^(a) processes the previous decoder hidden state (s_{t-1})
        # s_{t-1} has dimension h (hidden_size)
        self.W_a = nn.Linear(hidden_size, hidden_size)
        
        # U^(a) processes the encoder outputs (h_i)
        # h_i from Bi-LSTM has dimension 2h (2 * hidden_size)
        self.U_a = nn.Linear(hidden_size * 2, hidden_size)
        
        # v^T processes the tanh output to a single attention score
        # Using bias=False because the preceding tanh already centers the data
        self.v = nn.Linear(hidden_size, 1, bias=False)
        
    def forward(self, s_prev: torch.Tensor, encoder_outputs: torch.Tensor, src_mask: torch.Tensor = None) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            s_prev: Previous decoder hidden state, shape (batch_size, hidden_size)
            encoder_outputs: Outputs from Bi-LSTM, shape (batch_size, seq_len, 2 * hidden_size)
            src_mask: Boolean tensor (batch_size, seq_len). True/1 for valid tokens, False/0 for <pad>.
        
        Returns:
            c_t: Context vector, shape (batch_size, 2 * hidden_size)
            alpha_t: Attention weights, shape (batch_size, seq_len)
        """
        # Project Decoder State
        # We add the middle dimension (seq_len = 1) so it can broadcast over all encoder time steps.
        s_prev_proj = self.W_a(s_prev).unsqueeze(1) 
        
        # Project Encoder Outputs
        # Apply linear layer U_a: 1024 → 512
        enc_out_proj = self.U_a(encoder_outputs)
        
        # Calculate Attention Energies (e_{t,i})
        # Apply v -> shape: (batch_size, seq_len, 1) -> squeeze to (batch_size, seq_len)
        energy = self.v(torch.tanh(s_prev_proj + enc_out_proj)).squeeze(2)
        
        # Apply Masking 
        if src_mask is not None:
            energy = energy.masked_fill(src_mask == 0, -1e10)
            
        # Calculate Attention Weights 
        alpha_t = F.softmax(energy, dim=1)
        
        # 6. Calculate Context Vector 
        c_t = torch.bmm(alpha_t.unsqueeze(1), encoder_outputs).squeeze(1)
        
        return c_t, alpha_t

