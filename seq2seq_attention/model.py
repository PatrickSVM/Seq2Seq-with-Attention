import torch.nn as nn
import torch


class Encoder(nn.Module):
    def __init__(
        self, vocab_size, emb_dim, hidden_dim_enc, hidden_dim_dec,padding_idx,  num_layers=1,
    ):
        super(Encoder).__init__()

        self.padding_idx = padding_idx

        # Embedding layer (vocab_size, emb_dim)
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=emb_dim)

        # Bidirectional GRU
        self.gru = nn.GRU(
            input_size=emb_dim,
            hidden_size=hidden_dim_enc,
            num_layers=num_layers,
            bidirectional=True,
            batch_first=True,
            bias=True,
        )

        # Linear layer to produce summary of src sentence
        # Applied to last hidden state of each dirction
        self.src_summary = nn.Linear(hidden_dim_enc * 2, hidden_dim_dec)

    def forward(self, src, src_len):
        """
        Input
        src (batch_size, max(src_len))
        src_len (batch_size)

        Return
        src_summary (batch_size, hidden_dim_enc)
        all_hidden_unpacked (batch_size, max(src_len), 2*enc_hidden_dim)
        """

        # (batch_size, max(src_len), emb_dim)
        embedded_src = self.embedding(src)

        # Pack sequences
        src_len = src_len.to("cpu")

        packed_embedded_src = nn.utils.rnn.pack_padded_sequence(
            embedded_src, lengths=src_len, batch_first=True, enforce_sorted=True
        )

        # Compute hidden states from gru
        # all_hidden - all hidden states for each sequence and sequence position
        # packed_sequence, all_hidden.data (num_unpadded_seq_steps, 2*enc_hidden_dim)
        # last_hidden (2, batch_size, enc_hidden_dim) concatenation of last hidden state in both directions
        all_hidden, last_hidden = self.gru(packed_embedded_src)

        # Transform all_hidden back to unpacked sequence
        # (batch_size, max(src_len), 2*enc_hidden_dim)
        all_hidden_unpacked, _ = nn.utils.rnn.pad_packed_sequence(all_hidden)

        # Concatenate hidden state of the two directions on dim=1
        # (batch_size, 2*enc_hidden_dim)
        last_hidden = torch.cat([last_hidden[0, :, :], last_hidden[1, :, :]], dim=1)

        # Run through last layer to convert it to dec_hidden_dim dimension
        # and get a summary vector of the src sentence
        # Note: No activation function - could use tanh
        src_summary = self.src_summary(last_hidden)

        # Get padding mask 
        padding_mask = self.get_padding_mask(src)

        return src_summary, all_hidden_unpacked, padding_mask
    
    @torch.no_grad()
    def get_padding_mask(self, src):
        """ 
        Compute boool mask for whether padding token or not.
        """
        mask = src == self.padding_idx
        return mask




class Attention(nn.Module):
    def __init__(self, hidden_dim_enc, hidden_dim_dec, padded_src_len):
        super(Attention).__init__()
        
        self.padded_src_len = padded_src_len

        # Init alignment layer 1 & 2 to compute energy_ij = a(s_i-1, h_j)
        self.allignment_layer1 = nn.Linear((hidden_dim_enc * 2) + hidden_dim_dec, hidden_dim_dec)
        self.allignment_layer2 = nn.Linear(hidden_dim_dec, 1, bias=False)
        
        # Init softmax to compute attention weights from energy
        self.attention_weights = nn.Softmax(padded_src_len, dim=1)
        
    def forward(self, hidden_dec, hidden_enc, padding_mask):
        # hidden_dec: last hidden state from decoder (batch_size, hidden_dim_dec)
        # hidden_enc: hidden states from encoder for each seq-position (batch_size, padded_seq_len, 2*enc_hidden_dim)
        
        # Concat hidden state of decoder to each hidden state of the encoder-hidden-seq
        # attention_input (batch_size, padded_src_len, 2*enhidden+dec_hidden)
        hidden = hidden.unsqueeze(1).repeat(1, self.padded_src_len, 1)
        attention_input = torch.cat([hidden_enc, hidden_dec], dim=2)
        
        # Compute alignment score alpha_ij
        # Represents importance of src word j to predict trg word i
        # Computation by 2 layer FFN with tanh activation, hidden layer size quals hidden_dim_dec
        energy = self.allignment_layer1(attention_input)  # (batch_size, padded_src_len, hidden_dim_dec)
        energy = torch.tanh(energy)
        energy = self.allignment_layer2()  # (batch_size, padded_src_len, 1)

        # Squeeze energy to (batch_size, padded_src_len)
        energy = energy.squeeze(2)
        
        # Eliminate influence of padding tokens before softmaxing
        # by setting their energy to tiny negative num 
        energy[padding_mask] = -1e10

        # Compute attention weights for each j by softmaxing
        attention_weights = self.attention_weights(energy) 
    
        return attention_weights



class Decoder(nn.Module):
    def __init__(
        self, vocab_size, emb_dim, hidden_dim_enc, hidden_dim_dec, num_layers=1
    ):
        super(Encoder).__init__()

        # Embedding layer (vocab_size, emb_dim)
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=emb_dim)

        # Bidirectional GRU
        self.gru = nn.GRU(
            input_size=emb_dim,
            hidden_size=hidden_dim_enc,
            num_layers=num_layers,
            bidirectional=True,
            batch_first=True,
            bias=True,
        )

        # Linear layer to produce summary of src sentence
        # Applied to last hidden state of each dirction
        self.src_summary = nn.Linear(hidden_dim_enc * 2, hidden_dim_dec)

    def forward(self, src, src_len):
        pass
