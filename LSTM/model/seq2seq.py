import torch
import torch.nn as nn
import random
from model.encoder import EncoderBiLSTM
from model.decoder import DecoderLSTM

class Seq2Seq(nn.Module):
    def __init__(self, encoder: EncoderBiLSTM, decoder: DecoderLSTM, 
                src_pad_idx: int, device: torch.device):
        super(Seq2Seq, self).__init__()
        
        self.encoder = encoder
        self.decoder = decoder
        self.src_pad_idx = src_pad_idx
        self.device = device
        
    def create_src_mask(self, src: torch.Tensor) -> torch.Tensor:
        """
        Returns True for actual tokens and False for <pad> tokens
        """
        mask = (src != self.src_pad_idx)
        return mask.to(self.device)

    def forward(self, src: torch.Tensor, tgt: torch.Tensor, teacher_forcing_ratio: float = 0.5) -> torch.Tensor:
        """
        Args:
            src: Source sequences, 
            tgt: Target sequences, 
            teacher_forcing_ratio: Probability of using actual target as next input
            
        Returns:
            outputs: Tensor storing the predicted logits for each time step  (batch_size, tgt_seq_len, tgt_vocab_size)
        """
        batch_size = src.shape[0]
        tgt_len = tgt.shape[1]
        tgt_vocab_size = self.decoder.vocab_size
        
        # Tensor to store decoder outputs
        outputs = torch.zeros(batch_size, tgt_len, tgt_vocab_size).to(self.device)
        
        # Create source mask for attention
        src_mask = self.create_src_mask(src)
        
        # Pass source through encoder
        # hidden_state (s_0, c_0) shape: (1, batch_size, hidden_size)
        encoder_outputs, hidden_state = self.encoder(src)
        
        # Initialize Decoder Input
        # The first input to the decoder is the <sos> token
        input_step = tgt[:, 0].unsqueeze(1)
        
        # Decoding Loop
        # We start looping from 1 as 0 is <sos>
        for t in range(1, tgt_len):
            
            # Perform one step of the decoder
            logits, hidden_state, _ = self.decoder(
                input_step=input_step,
                hidden_state=hidden_state,
                encoder_outputs=encoder_outputs,
                src_mask=src_mask
            )
            
            # Place predictions into the outputs tensor
            outputs[:, t, :] = logits
            
            # Get the highest predicted token from the logits
            top1 = logits.max(1)[1]
            
            # Decide if we are going to use teacher forcing this step
            teacher_force = random.random() < teacher_forcing_ratio
            
            if teacher_force:
                input_step = tgt[:, t].unsqueeze(1)
            else:
                input_step = top1.unsqueeze(1)
                
        return outputs
