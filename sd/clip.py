import torch
from torch import nn
from torch.nn import functional as F
from attention import SelfAttention

'''
clip.py explained
'''

class CLIPEmbedding(nn.Module):
    '''
    
    '''
    def __init__(self, n_vocab, n_embdm, n_token):
        super().__init__()

        self.token_embedding = nn.Embedding(n_vocab, n_embdm)

        # a learnable weight matrix encodes the position information for each token
        self.position_embedding = nn.Parameter(torch.zeros((n_token, n_embdm)))
    
    def forward(self, tokens):

        # (batch_size, seq_length) -> (batch_size, seq_length, Dim)
        x = self.token_embedding(tokens)

        # (batch_size, seq_length) -> (batch_size, seq_length, Dim)
        x += self.position_embedding

        return x
    

class CLIPLayer(nn.Module):
    '''
    
    '''
    def __init__(self, n_head, n_embd):
        super().__init__()


        # Pre attention normalisation
        self.layernorm_1 = nn.LayerNorm(n_embd)
        # self attention
        self.attention = SelfAttention(n_head, n_embd)
        # Pre feed forward NN
        self.layernorm_2 = nn.LayerNorm(n_embd)
        # feed forward layer
        self.linear_1 = nn.Linear(n_embd, 4*n_embd)
        self.linear_2 = nn.Linear(4*n_embd, n_embd)

    def forward(self, x):
        # (batch_size, sequence_length, Dimension)
        residue = x

        ### self attention ## 
        # (Batch_Size, Seq_Len, Dim) -> (Batch_Size, Seq_Len, Dim)
        x = self.layernorm_1(x)
        # (Batch_Size, Seq_Len, Dim) -> (Batch_Size, Seq_Len, Dim)
        x  = self.attention(x, casual_mask = True)

        # (Batch_Size, Seq_Len, Dim) + (Batch_Size, Seq_Len, Dim) -> (Batch_Size, Seq_Len, Dim)
        x += residue

        ### feedforward layer ###

        residue = x
        # (Batch_Size, Seq_Len, Dim) -> (Batch_Size, Seq_Len, Dim)
        x = self.layernorm_2(x)

        # (Batch_Size, Seq_Len, Dim) -> (Batch_Size, Seq_Len, 4 * Dim)
        x = self.linear_1(x)

        # (Batch_Size, Seq_Len, 4 * Dim) -> (Batch_Size, Seq_Len, 4 * Dim)
        # quick GELU activation function
        x = x*torch.sigmoid(1.702*x)

        # (Batch_Size, Seq_Len, 4 * Dim) -> (Batch_Size, Seq_Len, Dim)
        x = self.linear_2(x)

        # (Batch_Size, Seq_Len, Dim) + (Batch_Size, Seq_Len, Dim) -> (Batch_Size, Seq_Len, Dim)
        x += residue

        return x

class CLIP(nn.Module):
    '''
    
    '''
    def __init__(self):
        super().__init__()

        # (vocabulary size, embedding size, sequence length)
        self.embedding = CLIPEmbedding(49408, 768, 77)

        self.layers = nn.ModuleList([
            CLIPLayer(12, 768) for i in range(12)
        ])

        self.layernorm = nn.LayerNorm(768)

    def forward(self, tokens: torch.LongTensor) -> torch.FloatTensor:

        tokens = tokens.type(torch.long)

        # (batch_size, sequence_length) -> (batch_size, sequence_length, Dimension = 768)
        state = self.embedding(tokens)

        # apply encoder layers similar to transformer encoder
        for layer in self.layers:
            # (batch_size, sequence_length, Dim) -> (batch_size, sequence_length, Dim)
            state = layer(state)
        # (batch_size, sequence_length, Dim) -> (batch_size, sequence_length, Dim)
        output = self.layernorm(state)

        return output