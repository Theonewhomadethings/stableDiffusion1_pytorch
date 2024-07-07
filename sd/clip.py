import torch
from torch import nn
from torch.nn import functional as F
from attention import SelfAttention

'''
    ### clip.py Explained ###

    CLIP (Contrastive Language–Image Pretraining) is a text encoder introduced by OpenAI in 2021. It is pretrained on a vast amount of internet data, enabling it to understand and generate meaningful embeddings for both text and images. CLIP operates in three primary stages: 
    1) Contrastive pretraining, 
    2) Creating a dataset classifier from label text, an*d 
    3) Zero-shot prediction. 

    In this implementation, we are not training CLIP from scratch but building its architecture and loading pretrained weights.

    CLIP (https://arxiv.org/abs/2103.00020) functions similarly to the encoder layer of a transformer architecture (https://arxiv.org/abs/1706.03762), utilizing self-attention, positional embeddings, layer normalizations, and linear neural network layers.

    The key innovation of CLIP is its ability to bridge the gap between text and image modalities, allowing for the generation of images from textual descriptions. This is achieved through a cosine similarity measure that aligns text and image embeddings, ensuring that correct text-image pairs have high similarity scores while incorrect pairs have low scores.

    The zero-shot prediction capability of CLIP makes it particularly useful for inference, as it can generalize to new tasks without additional training. CLIP is an integral component of text-to-image latent diffusion models, facilitating the generation of images based on textual prompts.

'''

class CLIPEmbedding(nn.Module):
    '''
    CLIPEmbedding creates token and position embeddings for the input text tokens.

    Attributes:
        token_embedding (nn.Embedding): Embedding layer that maps each token to its corresponding embedding vector.
        position_embedding (nn.Parameter): Learnable parameter that encodes positional information for each token.
    '''
    def __init__(self, n_vocab, n_embdm, n_token):
        '''
        Initializes the CLIPEmbedding with token and positional embeddings.

        Args:
            n_vocab (int): Size of the vocabulary.
            n_embd (int): Dimensionality of the embeddings.
            n_token (int): Maximum number of tokens in a sequence.
        '''
        super().__init__()

        self.token_embedding = nn.Embedding(n_vocab, n_embdm)

        # a learnable weight matrix encodes the position information for each token
        self.position_embedding = nn.Parameter(torch.zeros((n_token, n_embdm)))
    
    def forward(self, tokens):
        '''
        Forward pass to generate the embeddings for the input tokens.

        Args:
            tokens (torch.Tensor): Input tensor containing token indices with shape (batch_size, seq_length).

        Returns:
            torch.Tensor: Embedding tensor with shape (batch_size, seq_length, n_embd).
        '''

        # (batch_size, seq_length) -> (batch_size, seq_length, Dim)
        x = self.token_embedding(tokens)

        # (batch_size, seq_length) -> (batch_size, seq_length, Dim)
        x += self.position_embedding

        return x
    

class CLIPLayer(nn.Module):
    '''
    CLIPLayer represents a single transformer encoder layer used in CLIP, consisting of 
    self-attention and feed-forward neural network sublayers with layer normalization.

    Attributes:
        layernorm_1 (nn.LayerNorm): Layer normalization applied before the self-attention mechanism.
        attention (SelfAttention): Self-attention mechanism.
        layernorm_2 (nn.LayerNorm): Layer normalization applied before the feed-forward neural network.
        linear_1 (nn.Linear): First linear layer in the feed-forward network, expanding the dimensionality.
        linear_2 (nn.Linear): Second linear layer in the feed-forward network, reducing the dimensionality.
    '''
    def __init__(self, n_head, n_embd):
        '''
        Initializes the CLIPLayer with specified number of heads for self-attention and embedding dimension.

        Args:
            n_head (int): Number of attention heads.
            n_embd (int): Dimensionality of the embeddings.
        '''
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
        '''
        Forward pass through the CLIP layer.

        Args:
            x (torch.Tensor): Input tensor with shape (batch_size, sequence_length, n_embd).

        Returns:
            torch.Tensor: Output tensor with the same shape, incorporating attention and feed-forward transformations.
        '''
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
    CLIP model comprising an embedding layer and a series of transformer encoder layers.

    Attributes:
        embedding (CLIPEmbedding): Embedding layer for tokens and positional encodings.
        layers (nn.ModuleList): List of transformer encoder layers.
        layernorm (nn.LayerNorm): Final layer normalization applied to the output.
    '''
    def __init__(self):
        '''
        Initializes the CLIP model with predefined vocabulary size, embedding dimension, and sequence length.
        '''
        super().__init__()

        # (vocabulary size, embedding size, sequence length)
        self.embedding = CLIPEmbedding(49408, 768, 77)

        self.layers = nn.ModuleList([
            CLIPLayer(12, 768) for i in range(12)
        ])

        self.layernorm = nn.LayerNorm(768)

    def forward(self, tokens: torch.LongTensor) -> torch.FloatTensor:
        '''
        Forward pass through the CLIP model.

        Args:
            tokens (torch.LongTensor): Input tensor containing token indices with shape (batch_size, seq_length).

        Returns:
            torch.FloatTensor: Output tensor with shape (batch_size, sequence_length, embedding_dim).
        '''
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
