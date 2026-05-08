import torch
import torch.nn as nn

class EncoderBiLSTM(nn.Module):
    def __init__(self, vocab_size: int, embed_size: int = 256, hidden_size: int = 512, 
                 num_layers: int = 1, dropout_prob: float = 0.3, padding_idx: int = None):
        super(EncoderBiLSTM, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.embedding = nn.Embedding(vocab_size, embed_size, padding_idx=padding_idx)
        self.dropout = nn.Dropout(dropout_prob)
        
        # Bidirectional LSTM
        self.lstm = nn.LSTM(
            input_size=embed_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bidirectional=True,
            batch_first=True,
            # PyTorch expects dropout=0 if num_layers == 1. 
            # We handle embedding dropout separately above.
            dropout=dropout_prob if num_layers > 1 else 0 
        )
        
        # Projection layers 
        self.w_init_h = nn.Linear(hidden_size * 2, hidden_size)
        
        self.w_init_c = nn.Linear(hidden_size * 2, hidden_size)

    def forward(self, src: torch.Tensor) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """
        Args:
            src: Source sequences, shape (batch_size, src_seq_len)
            
        Returns:
            encoder_outputs: h_i = [h_forward_i; h_backward_i], shape (batch_size, src_seq_len, 2 * hidden_size)
            decoder_init_state: Tuple of (h_0, c_0), both shape (1, batch_size, hidden_size)
        """
        # Embed and apply dropout
        embedded = self.dropout(self.embedding(src))
        
        # Pass through Bi-LSTM
        # encoder_outputs shape: (batch_size, src_seq_len, 2 * hidden_size)
        # h_n shape: (num_layers * 2, batch_size, hidden_size)
        # c_n shape: (num_layers * 2, batch_size, hidden_size)
        encoder_outputs, (h_n, c_n) = self.lstm(embedded)
        
        # Decoder Initialization
        forward_hidden = h_n[-2, :, :]  
        backward_hidden = h_n[-1, :, :] 
        
        # Concatenate
        concat_hidden = torch.cat((forward_hidden, backward_hidden), dim=1)
        
        # Apply projection and tanh to get initial hidden state for decoder
        s_0_h = torch.tanh(self.w_init_h(concat_hidden))
        
        # same for  cell state to keep PyTorch happy
        forward_cell = c_n[-2, :, :]
        backward_cell = c_n[-1, :, :]
        concat_cell = torch.cat((forward_cell, backward_cell), dim=1)
        s_0_c = torch.tanh(self.w_init_c(concat_cell))
        
        # Reshape to (num_layers=1, batch_size, hidden_size) for the Unidirectional Decoder
        s_0_h = s_0_h.unsqueeze(0)
        s_0_c = s_0_c.unsqueeze(0)
        
        decoder_init_state = (s_0_h, s_0_c)
        
        return encoder_outputs, decoder_init_state

