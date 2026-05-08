import torch
import torch.nn as nn
import math

class InputEmbedding(nn.Module):
    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        return self.embedding(x) * math.sqrt(self.d_model)
    
class PositionalEncoding(nn.Module):
    def __init__(self, d_model : int, seq_len: int, dropout: float) -> None:
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)


        pe = torch.zeros(seq_len, d_model)

        #indices
        positions = torch.arange(0, seq_len , dtype = float).unsqueeze(1)
        # denominator 
        div_term = torch.exp(torch.arange(0,d_model,2).float() * (-math.log(10000.0)/d_model))

        # applying sin for even indecies and cos for odd ones
        pe[:,0::2] = torch.sin(positions * div_term)
        pe[:,1::2] = torch.cos(positions * div_term)
        
        #adding batch dim for broadcasting later
        pe = pe.unsqueeze(0)

        # to keep it extractable with the model 
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
            adds PE to reuqired sentence 
        """
        x = x + (self.pe[:,:x.shape[1],:]).requires_grad_(False)
        return self.dropout(x)


class LayerNormalization(nn.Module):

    def __init__(self, eps: float = 10**-6) -> None:
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(1)) # multiplitve trainable parameter
        self.bias = nn.Parameter(torch.zeros(1)) # additive trainable parameter

    def forward(self, x):
        
        """
            LayerNorm(x) = a * ((x - μ) / √(σ² + ε)) + β
        """
        mean = x.mean(dim = -1, keepdim = True)
        std = x.std(dim = -1, keepdim = True ,unbiased=False)
        return self.alpha * (x - mean) / (std + self.eps) + self.bias
        

class FeedForwardBlock(nn.Module):

    def __init__(self,d_model: int, d_ff: int, dropout: float, activation = 'relu') -> None:
        super().__init__()
        self.linear1 = nn.Linear(d_model,d_ff) #W1 and B2
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff, d_model)

        if activation == 'gelu':
            self.activation = nn.GELU()  # or use F.gelu in forward
        elif activation == 'relu':
            self.activation = nn.ReLU()
        else:
            raise ValueError(f"Unknown activation: {activation}")


    def forward(self,x):
        #(Batch,seq_len, d_model) --> (Batch,seq_len, d_ff) --> (Batch,seq_len, d_model)
        x = self.linear1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x


class MultiHeadAttentionBlock(nn.Module):
    def __init__(self, d_model : int, h: int, dropout:float) -> None:
            super().__init__()
            self.d_model = d_model
            self.h = h
            assert d_model % h == 0, "Model Dimension is not divisible by h"

            self.d_k = d_model // h


            self.w_q = nn.Linear(d_model,d_model) # Wq
            self.w_k = nn.Linear(d_model,d_model) # Wk
            self.w_v = nn.Linear(d_model,d_model) # Wv

            self.w_o = nn.Linear(d_model,d_model) # Wo

            self.dropout = nn.Dropout(dropout)
    @staticmethod 

    def attention(query, key, value, mask, dropout: nn.Dropout):
        d_k = query.shape[-1] 

        #(Batch, h, seq_len, d_k) --> (Batch, h ,seq_len, seq_len)
        attention_scores =  (query @ key.transpose(-2,-1)) / math.sqrt(d_k)
        
        if mask is not None:
            attention_scores.masked_fill(~mask, -1e9)
        attention_scores = attention_scores.softmax(dim = -1)
        if dropout is not None:
            attention_scores = dropout(attention_scores)

        return (attention_scores @ value), attention_scores

    def forward(self,q ,k ,v, mask: None):
        query = self.w_q(q) #(Batch,seq_len, d_model) --> (Batch,seq_len, d_model)
        key = self.w_k(k)   #(Batch,seq_len, d_model) --> (Batch,seq_len, d_model)
        value = self.w_v(v) #(Batch,seq_len, d_model) --> (Batch,seq_len, d_model)


        #(Batch,seq_len, d_model) --> (Batch, h ,seq_len, d_k)
        query =query.view(query.shape[0],query.shape[1], self.h, self.d_k).transpose(1,2)
        key = key.view(key.shape[0], key.shape[1], self.h, self.d_k).transpose(1,2)
        value = value.view(value.shape[0], value.shape[1], self.h, self.d_k).transpose(1,2)
        x, self.attention_scores = MultiHeadAttentionBlock.attention(query,key,value, mask, self.dropout)
        
        #(Batch, h ,seq_len, d_k) --> (Batch, seq_len, h, d_k) --> (Batch, seq_len, d_model(dv))
        x = x.transpose(1,2).contiguous().view(x.shape[0], -1, self.h * self.d_k)
        return self.w_o(x)
    

class ResidualConnection(nn.Module):
    def __init__(self, dropout: float) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNormalization()

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))

class EncoderBlock(nn.Module):
    def __init__(self, self_attention_block : MultiHeadAttentionBlock, feed_forward_block: FeedForwardBlock, dropout : float) -> None:
        super().__init__()
        self.self_attention_block = self_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connection = nn.ModuleList(ResidualConnection(dropout) for _ in range(2))
    
    def forward(self, x ,src_mask):
        x = self.residual_connection[0](x, lambda x: self.self_attention_block(x,x,x,src_mask))
        x = self.residual_connection[1](x,self.feed_forward_block)
        return x

class Encoder(nn.Module):
    def __init__(self, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization()

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x,mask)

        return self.norm(x)
    

class DecoderBlock(nn.Module):
    def __init__(self, self_attention_block : MultiHeadAttentionBlock, cross_attention_block: MultiHeadAttentionBlock, feed_forward_block: FeedForwardBlock, dropout: float) -> None:
        super().__init__()
        self.self_attention_block = self_attention_block
        self.cross_attention_block = cross_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList(ResidualConnection(dropout) for _ in range(3))

    def forward(self, x, encoder_output,src_mask, trgt_mask):
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x , x , trgt_mask))
        x = self.residual_connections[1](x, lambda x: self.cross_attention_block(x, encoder_output,encoder_output, src_mask))
        x = self.residual_connections[2](x, self.feed_forward_block)
        return x
    

class Decoder(nn.Module):
    def __init__(self, layers : nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization()

    def forward(self, x, encoder_output, src_mask, trgt_mask):
        for layer in self.layers: 
            x = layer(x, encoder_output, src_mask, trgt_mask)
        
        return self.norm(x)
    
class ProjectionLayer(nn.Module):
    def __init__(self, d_model: int , vocab_size: int) -> None:
        super().__init__()
        self.proj = nn.Linear(d_model,vocab_size)

    def forward(self,x):
        # (batch, Seq_len, d_model) -> (batch, Seq_len, Vocab_size)
        return self.proj(x)        # raw logits, no log_softmax
    

class Transformer(nn.Module):
    def __init__(self, encoder: Encoder, decoder: Decoder, src_embed: InputEmbedding, trgt_embed: InputEmbedding, src_pos: PositionalEncoding, trgt_pos: PositionalEncoding, projection_layer: ProjectionLayer) -> None: 
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = trgt_embed
        self.src_pos = src_pos 
        self.tgt_pos = trgt_pos
        self.projection_layer = projection_layer

    def encode(self, src, src_mask):
        src = self.src_embed(src)
        src = self.src_pos(src)
        return self.encoder(src, src_mask)
    def decode(self,tgt, encoder_output, tgt_mask, src_mask):
        tgt =  self.tgt_embed(tgt)
        tgt = self.tgt_pos(tgt)
        return self.decoder(tgt, encoder_output, src_mask, tgt_mask)
    
    def project(self,x):
        return self.projection_layer(x)
    

def build_transformer(src_vocab_size: int, tgt_vocab_size: int, src_seq_len: int, tgt_seq_len: int, d_model: int = 512, N: int = 6, h: int =8, dropout: float = 0.1, d_ff: int = 2048) -> Transformer:
    # create embedding layers 
    src_embed = InputEmbedding(d_model,src_vocab_size)
    tgt_embed = InputEmbedding(d_model,tgt_vocab_size)

    # create the positional encoding layers 
    src_pos = PositionalEncoding(d_model, src_seq_len, dropout)
    tgt_pos = PositionalEncoding(d_model, tgt_seq_len, dropout)

    # create the encoder blocks

    encoder_layers = []

    for _ in range(N):
        encoder_self_att_block = MultiHeadAttentionBlock(d_model, h, dropout)
        feed_forward_block = FeedForwardBlock(d_model,d_ff,dropout)
        encoder_layer = EncoderBlock(encoder_self_att_block, feed_forward_block, dropout)
        encoder_layers.append(encoder_layer)

    # create the encoder blocks

    decoder_layers = []

    for _ in range(N):
        decoder_self_att_block = MultiHeadAttentionBlock(d_model, h, dropout)
        decoder_cross_att_block = MultiHeadAttentionBlock(d_model, h, dropout)
        feed_forward_block = FeedForwardBlock(d_model,d_ff,dropout)
        decoder_layer = DecoderBlock(decoder_self_att_block,decoder_cross_att_block, feed_forward_block, dropout)
        decoder_layers.append(decoder_layer)

    # create the encoder and decoder
    encoder = Encoder(nn.ModuleList(encoder_layers))
    decoder = Decoder(nn.ModuleList(decoder_layers))

    # final projection layer 
    projection_layer = ProjectionLayer(d_model, tgt_vocab_size)

    # create tranformer
    transformer = Transformer(encoder,decoder,src_embed,tgt_embed,src_pos,tgt_pos,projection_layer)


    # initializing the parameters 
    for p in transformer.parameters():
        if p.dim() > 1: 
            nn.init.xavier_uniform_(p)

    return transformer
    
