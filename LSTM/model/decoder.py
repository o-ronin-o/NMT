import torch
import torch.nn as nn
from model.attention import AdditiveAttention

class DecoderLSTM(nn.Module):
    def __init__(self, vocab_size: int, embed_size: int = 256, hidden_size: int = 512, 
                num_layers: int = 1, dropout_prob: float = 0.3, padding_idx: int = None):
        super(DecoderLSTM, self).__init__()
        
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        
        # 1. Embedding Layer for Target Language (English)
        self.embedding = nn.Embedding(vocab_size, embed_size, padding_idx=padding_idx)
        self.dropout = nn.Dropout(dropout_prob)
        
        # 2. Additive Attention Mechanism
        self.attention = AdditiveAttention(hidden_size)
        
        # 3. Unidirectional LSTM
        # Input to LSTM is [y_{t-1}; c_t]
        lstm_input_size = embed_size + (2 * hidden_size)
        
        self.lstm = nn.LSTM(
            input_size=lstm_input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bidirectional=False, # unidirectional for Decoder
            batch_first=True,
            dropout=dropout_prob if num_layers > 1 else 0
        )
        
        # 4. Output Layer
        # logits_t = W^{(out)} [s_t; c_t; y_{t-1}]
        # s_t size: hidden_size (512)
        # c_t size: 2 * hidden_size (1024)
        # y_{t-1} size: embed_size (256)
        # = 512 + 1024 + 256 = 1792
        output_input_size = hidden_size + (2 * hidden_size) + embed_size
        self.w_out = nn.Linear(output_input_size, vocab_size)

    def forward(self, input_step: torch.Tensor, hidden_state: tuple[torch.Tensor, torch.Tensor], 
                encoder_outputs: torch.Tensor, src_mask: torch.Tensor = None) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        """
        Executes a SINGLE step of the decoder (for time step t).
        
        Args:
            input_step: The target token for the current step (y_{t-1}), shape (batch_size, 1)
            hidden_state: Tuple (h_{t-1}, c_{t-1}) from previous decoder step.
                          Both shape (num_layers, batch_size, hidden_size)
            encoder_outputs: Context from encoder, shape (batch_size, seq_len, 2 * hidden_size)
            src_mask: Mask for encoder outputs to ignore padding.
            
        Returns:
            logits: Predictions for next token, shape (batch_size, vocab_size)
            hidden_state: New (h_t, c_t) for the next step
            alpha_t: Attention weights for visualization later
        """
        # Embed the input token
        embedded = self.dropout(self.embedding(input_step))
        
        # Extract s_{t-1} 
        s_prev = hidden_state[0][-1] # Take the top layer's hidden state
        
        # Compute Attention
        c_t, alpha_t = self.attention(s_prev, encoder_outputs, src_mask)
        
        # Reshape c_t to combine with embedded input 
        c_t_unsqueeze = c_t.unsqueeze(1)
        
        # Concatenate input and context vector for LSTM
        lstm_input = torch.cat((embedded, c_t_unsqueeze), dim=2)
        
        # Pass through LSTM
        # hidden_state is updated to (h_t, c_t)
        output, hidden_state = self.lstm(lstm_input, hidden_state)
        
        #  Concatenate for Final Output Projection
        # final_concat shape: (batch_size, hidden_size + 2*hidden_size + embed_size)
        final_concat = torch.cat((output.squeeze(1), c_t, embedded.squeeze(1)), dim=1)
        
        # 6. Predict next word 
        logits = self.w_out(final_concat)
        
        return logits, hidden_state, alpha_t
