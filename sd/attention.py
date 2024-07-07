import torch
from torch import nn
from torch.nn import functional as F
import math

'''
    ### attention.py Explained ###

    This script contains the implementation of SelfAttention and CrossAttention mechanisms, which are crucial for the operation of various neural network architectures, including transformers and models used in stable diffusion.

    - **SelfAttention**: This mechanism allows the model to focus on different parts of a single sequence. It computes attention weights using the same sequence as the source of queries, keys, and values. The implementation is based on the transformer model introduced in "Attention is All You Need" (https://arxiv.org/abs/1706.03762).
    
    - **CrossAttention**: This mechanism enables the model to focus on different parts of one sequence while processing another. It uses different sequences for queries, keys, and values, making it suitable for tasks that require merging information from multiple modalities. The concept was formalized in papers such as "Cross-Attention is All You Need: Adapting Pretrained Transformers for Machine Translation" (https://arxiv.org/abs/2104.08771).
'''

class SelfAttention(nn.Module):
    '''
    SelfAttention layer for capturing relationships within a single sequence.

    This mechanism allows the model to attend to different positions within the same sequence to capture dependencies and contextual information. 
    It computes attention scores for each pair of positions in the sequence using query, key, and value projections. 
    These scores are used to weight the values and produce a context-aware representation of each position.


    Args:
        n_heads (int): Number of attention heads.
        d_embed (int): Dimension of the embedding.
        in_proj_bias (bool): Whether to include bias in the input projection.
        out_proj_bias (bool): Whether to include bias in the output projection.

    Attributes:
        in_proj (nn.Linear): Linear layer for projecting inputs to query, key, and value.
        out_proj (nn.Linear): Linear layer for projecting the output.
        n_heads (int): Number of attention heads.
        d_head (int): Dimension of each attention head.
    '''
    def __init__(self, n_heads, d_embed, in_proj_bias=True, out_proj_bias=True):
        super().__init__()

        self.in_proj = nn.Linear(d_embed, 3*d_embed, bias=in_proj_bias)

        self.out_proj = nn.Linear(d_embed, bias=out_proj_bias)
       
        self.n_heads = n_heads

        self.d_head = d_embed // n_heads

    def forward(self, x, casual_mask=False):
        '''
        Forward pass for self-attention.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_length, d_embed).
            casual_mask (bool): Whether to apply a causal mask for autoregressive tasks.

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_length, d_embed).
        '''
        #x: (batch_size, seq_length, Dimensions)

        input_shape = x.shape

        batch_size, sequence_length, d_embed = input_shape

        interim_shape = (batch_size, sequence_length, self.n_heads, self.d_head)

        # (Batch_Size, Seq_Len, Dim) -> (Batch_Size, Seq_Len, Dim * 3) -> 3 tensor of shape (Batch_Size, Seq_Len, Dim)        
        q, k, v = self.in_proj(x).chunk(3, dim=1)

        # (Batch_Size, Seq_Len, Dim) -> (Batch_Size, Seq_Len, H, Dim / H) -> (Batch_Size, H, Seq_Len, Dim / H)
        q = q.view(interim_shape).transpose(1,2)
        k = k.view(interim_shape).transpose(1,2)
        v = v.view(interim_shape).transpose(1,2)

        # (Batch_Size, H, Seq_Len, Dim / H) @ (Batch_Size, H, Dim / H, Seq_Len) -> (Batch_Size, H, Seq_Len, Seq_Len)
        weight = q @ k.transpose(-1, -2)

        if casual_mask:
            # mask where the upper traingle is 1
            mask = torch.ones_like(weight, dtype=torch.bool).triu(1)

            # fill the upper triangle with -inf
            weight.masked_fill_(mask, -torch.inf)

        # Divide by d_k (Dim / H). 
        # (Batch_Size, H, Seq_Len, Seq_Len) -> (Batch_Size, H, Seq_Len, Seq_Len)
        weight /= math.sqrt(self.d_head)

        # (Batch_Size, H, Seq_Len, Seq_Len) -> (Batch_Size, H, Seq_Len, Seq_Len)
        weight = F.softmax(weight, dim =-1)

        # (Batch_Size, H, Seq_Len, Seq_Len) @ (Batch_Size, H, Seq_Len, Dim / H) -> (Batch_Size, H, Seq_Len, Dim / H)
        output = weight @ v

        # (Batch_Size, H, Seq_Len, Dim / H) -> (Batch_Size, Seq_Len, H, Dim / H)
        output = output.transpose(1,2)

        # (Batch_Size, Seq_Len, H, Dim / H) -> (Batch_Size, Seq_Len, Dim)
        output = output.reshape(input_shape)

        # (Batch_Size, Seq_Len, Dim) -> (Batch_Size, Seq_Len, Dim)
        output = self.out_proj(output) 

        # (Batch_Size, Seq_Len, Dim)
        return output


class CrossAttention(nn.Module):
    '''
    CrossAttention layer for capturing relationships between two sequences.

    CrossAttention layer for capturing relationships between two sequences.

    This mechanism allows the model to attend to positions in one sequence (context) while processing another sequence (latent). 
    It computes attention scores using query projections from the latent sequence and key-value projections from the context sequence. 
        These scores are used to weight the values and produce a context-aware representation of the latent sequence.


    Args:
        n_heads (int): Number of attention heads.
        d_embed (int): Dimension of the embedding.
        d_cross (int): Dimension of the cross sequence embedding.
        in_proj_bias (bool): Whether to include bias in the input projection.
        out_proj_bias (bool): Whether to include bias in the output projection.

    Attributes:
        q_proj (nn.Linear): Linear layer for projecting queries.
        k_proj (nn.Linear): Linear layer for projecting keys.
        v_proj (nn.Linear): Linear layer for projecting values.
        out_proj (nn.Linear): Linear layer for projecting the output.
        n_heads (int): Number of attention heads.
        d_head (int): Dimension of each attention head.
    '''
    def __init__(self, n_heads, d_embed, d_cross, in_proj_bias=True, out_proj_bias=True):
        super().__init__() 
        self.q_proj   = nn.Linear(d_embed, d_embed, bias=in_proj_bias)
        self.k_proj   = nn.Linear(d_cross, d_embed, bias=in_proj_bias)
        self.v_proj   = nn.Linear(d_cross, d_embed, bias=in_proj_bias)
        self.out_proj = nn.Linear(d_embed, d_embed, bias=out_proj_bias)

        self.n_heads = n_heads
        self.d_head = d_embed // n_heads

    def forward(self, x, y):
        '''
        Forward pass for cross-attention.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_length_q, d_embed).
            y (torch.Tensor): Context tensor of shape (batch_size, seq_length_kv, d_cross).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_length_q, d_embed).
        '''
        # x (latent): # (Batch_Size, Seq_Len_Q, Dim_Q)
        # y (context): # (Batch_Size, Seq_Len_KV, Dim_KV) = (Batch_Size, 77, 768)

        input_shape = x.shape
        batch_size, sequence_length, d_embed = input_shape
        # Divide each embedding of Q into multiple heads such that d_heads * n_heads = Dim_Q
        interim_shape = (batch_size, -1, self.n_heads, self.d_head)
        
        # (Batch_Size, Seq_Len_Q, Dim_Q) -> (Batch_Size, Seq_Len_Q, Dim_Q)
        q = self.q_proj(x)
        # (Batch_Size, Seq_Len_KV, Dim_KV) -> (Batch_Size, Seq_Len_KV, Dim_Q)
        k = self.k_proj(y)
        # (Batch_Size, Seq_Len_KV, Dim_KV) -> (Batch_Size, Seq_Len_KV, Dim_Q)
        v = self.v_proj(y)

        # (Batch_Size, Seq_Len_Q, Dim_Q) -> (Batch_Size, Seq_Len_Q, H, Dim_Q / H) -> (Batch_Size, H, Seq_Len_Q, Dim_Q / H)
        q = q.view(interim_shape).transpose(1, 2) 
        # (Batch_Size, Seq_Len_KV, Dim_Q) -> (Batch_Size, Seq_Len_KV, H, Dim_Q / H) -> (Batch_Size, H, Seq_Len_KV, Dim_Q / H)
        k = k.view(interim_shape).transpose(1, 2) 
        # (Batch_Size, Seq_Len_KV, Dim_Q) -> (Batch_Size, Seq_Len_KV, H, Dim_Q / H) -> (Batch_Size, H, Seq_Len_KV, Dim_Q / H)
        v = v.view(interim_shape).transpose(1, 2) 
        
        # (Batch_Size, H, Seq_Len_Q, Dim_Q / H) @ (Batch_Size, H, Dim_Q / H, Seq_Len_KV) -> (Batch_Size, H, Seq_Len_Q, Seq_Len_KV)
        weight = q @ k.transpose(-1, -2)
        
        # (Batch_Size, H, Seq_Len_Q, Seq_Len_KV)
        weight /= math.sqrt(self.d_head)
        
        # (Batch_Size, H, Seq_Len_Q, Seq_Len_KV)
        weight = F.softmax(weight, dim=-1)
        
        # (Batch_Size, H, Seq_Len_Q, Seq_Len_KV) @ (Batch_Size, H, Seq_Len_KV, Dim_Q / H) -> (Batch_Size, H, Seq_Len_Q, Dim_Q / H)
        output = weight @ v
        
        # (Batch_Size, H, Seq_Len_Q, Dim_Q / H) -> (Batch_Size, Seq_Len_Q, H, Dim_Q / H)
        output = output.transpose(1, 2).contiguous()
        
        # (Batch_Size, Seq_Len_Q, H, Dim_Q / H) -> (Batch_Size, Seq_Len_Q, Dim_Q)
        output = output.view(input_shape)
        
        # (Batch_Size, Seq_Len_Q, Dim_Q) -> (Batch_Size, Seq_Len_Q, Dim_Q)
        output = self.out_proj(output)

        # (Batch_Size, Seq_Len_Q, Dim_Q)
        return output
