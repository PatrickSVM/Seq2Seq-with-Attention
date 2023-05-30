import torch.nn as nn
import torch
import random


def weighted_sum(H, W):
    """
    Computes weighted sum of hidden states at each seq step
    according to weights for each batch and step in W.

    H: (batch_size, seq_len, hidden_dim)
    W: (batch_size, seq_len)

    Return:
    weigted_sum (batch_size, seq_len)
    """
    weighted_sum = (H * W.unsqueeze(2)).sum(dim=1)
    return weighted_sum


class Encoder(nn.Module):
    def __init__(
        self,
        vocab_size,
        emb_dim,
        hidden_dim_enc,
        hidden_dim_dec,
        padding_idx,
        dropout,
        num_layers=1,
    ):
        super(Encoder, self).__init__()

        self.padding_idx = padding_idx

        # Embedding layer (vocab_size, emb_dim)
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=emb_dim)

        # Init dropout
        self.dropout = nn.Dropout(dropout)

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
        src_summary (batch_size, hidden_dim_dec)
        all_hidden_unpacked (batch_size, max(src_len), 2*enc_hidden_dim)
        """

        # (batch_size, max(src_len), emb_dim)
        embedded_src = self.dropout(self.embedding(src))

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
        all_hidden_unpacked, _ = nn.utils.rnn.pad_packed_sequence(
            all_hidden, batch_first=True
        )

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
    """
    Attention mechanism for RNN-based seq2seq encoder decoder
    networks based on the paper by Bahdanau et al.

    https://arxiv.org/abs/1409.0473
    """

    def __init__(
        self,
        hidden_dim_enc,
        hidden_dim_dec,
    ):
        super(Attention, self).__init__()

        # Init alignment layer 1 & 2 to compute energy_ij = a(s_i-1, h_j)
        self.alignment_layer1 = nn.Linear(
            (hidden_dim_enc * 2) + hidden_dim_dec, hidden_dim_dec
        )
        self.alignment_layer2 = nn.Linear(hidden_dim_dec, 1, bias=False)

        # Init softmax to compute attention weights from energy
        self.attention_weights = nn.Softmax(dim=1)

    def forward(self, hidden_dec, hidden_enc, padding_mask):
        # hidden_dec: last hidden state from decoder (batch_size, hidden_dim_dec)
        # hidden_enc: hidden states from encoder for each seq-position (batch_size, padded_seq_len, 2*enc_hidden_dim)

        # Concat hidden state of decoder to each hidden state of the encoder-hidden-seq
        # attention_input (batch_size, padded_src_len, 2*enhidden+dec_hidden)
        padded_src_len = hidden_enc.shape[1]
        hidden_dec = hidden_dec.unsqueeze(1).repeat(1, padded_src_len, 1)

        attention_input = torch.cat([hidden_enc, hidden_dec], dim=2)

        # Compute alignment score alpha_ij
        # Represents importance of src word j to predict trg word i
        # Computation by 2 layer FFN with tanh activation, hidden layer size quals hidden_dim_dec
        energy = self.alignment_layer1(
            attention_input
        )  # (batch_size, padded_src_len, hidden_dim_dec)
        energy = torch.tanh(energy)
        energy = self.alignment_layer2(energy)  # (batch_size, padded_src_len, 1)

        # Squeeze energy to (batch_size, padded_src_len)
        energy = energy.squeeze(2)

        # Eliminate influence of padding tokens before softmaxing
        # by setting their energy to huge negative num
        energy[padding_mask] = -1e10

        # Compute attention weights (batch_size, padded_seq_len) for each j by softmaxing
        attention_weights = self.attention_weights(energy)

        return attention_weights


class UniformAttention(nn.Module):
    """
    Fixed uniform attentino layer - every timestep
    will be weighted equally.
    """

    def __init__(
        self,
    ):
        super(UniformAttention, self).__init__()

        # Init softmax to compute probability distribution over tensor of ones
        self.uniform_weights = nn.Softmax(dim=1)

    def forward(self, hidden_dec, hidden_enc, padding_mask):

        # Compute attention weights (batch_size, padded_seq_len) for each j by softmaxing
        attention_weights = self.uniform_weights(
            torch.ones(size=(hidden_enc.shape[0], hidden_enc.shape[1]))
        )

        return attention_weights


class Decoder(nn.Module):
    def __init__(
        self,
        hidden_dim_enc,
        hidden_dim_dec,
        emb_dim_trg,
        dropout,
        vocab_size_trg,
        num_layers=1,
    ):
        super(Decoder, self).__init__()

        # Safe the target vocab size
        self.vocab_size_trg = vocab_size_trg

        # Init target embedding layer (vocab_size, emb_dim)
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size_trg, embedding_dim=emb_dim_trg
        )

        # Init dropout
        self.dropout = nn.Dropout(dropout)

        # Init unidirectional decoder GRU to get next decoder hidden state
        # s_i = gru(s_i-1, y_i-1, c_i)
        self.gru = self.gru = nn.GRU(
            input_size=hidden_dim_dec + (2 * hidden_dim_enc) + emb_dim_trg,
            hidden_size=hidden_dim_dec,
            num_layers=num_layers,
            bidirectional=False,
            batch_first=True,
            bias=True,
        )

        # Init target output layer g(s_i, y_i-1, c_i)
        self.output_layer = nn.Linear(
            in_features=hidden_dim_dec + (2 * hidden_dim_enc) + emb_dim_trg,
            out_features=vocab_size_trg,
        )

    def forward(self, s_bef, y_bef, c_i):
        """
        s_bef (batch_size, hidden_dim_dec): hidden state of decoder timestep before
        y_bef (batch_size, 1): target word before (predicted/gold-standard)
        c_i (batch_size, 2*hidden_dim_enc):
        """
        # Embed y_bef to (batch_size, embed_dim)
        y_bef_embed = self.dropout(self.embedding(y_bef.squeeze()))

        # Concat (s_i-1, y_i-1, c_i) to input for GRU
        # (batch_size, hidden_dim_dec+(2*hidden_dim_enc)+emb_dim_trg)
        if len(y_bef_embed.shape) == 1:
            y_bef_embed = y_bef_embed.unsqueeze(0)

        gru_input = torch.cat(
            [s_bef.squeeze(dim=-1), y_bef_embed.squeeze(dim=-1), c_i.squeeze(dim=-1)],
            dim=1,
        )

        # Unsqueeze in dim 1 to (batch_size, 1, feat_dim)
        gru_input = gru_input.unsqueeze(dim=1)

        # Compute s_i as hidden state for next output
        # (batch_size, hiddem_dim_trg)
        _, s_current = self.gru(gru_input)

        # Concat (s_i, y_i-1, c_i) to input for target layer
        # (batch_size, feat_dim)
        trg_input = torch.cat(
            [
                s_current.squeeze(dim=0),
                y_bef_embed.squeeze(dim=-1),
                c_i.squeeze(dim=-1),
            ],
            dim=1,
        )

        # Compute forward pass of output model - word logits
        # (batch_size, trg_vocab_size)
        next_output = self.output_layer(trg_input)

        return next_output, s_current


class Seq2Seq_Architecture_with_Att(nn.Module):
    def __init__(
        self,
        encoder,
        decoder,
        attention,
        device="cuda",
        init_token_idx=2,
    ):
        super(Seq2Seq_Architecture_with_Att, self).__init__()

        # Init all parts
        self.encoder = encoder
        self.attention = attention
        self.decoder = decoder
        self.init_token_idx = init_token_idx
        self.device = device

    def forward(self, src_batch, trg_batch, src_len, teacher_forcing):
        # Compute entire forward pass on src batch for all steps
        # h_dec (batch_size, hidden_dim_dec): first hidden state for decoder
        # h_enc (batch_size, seq_len, hidden_dim_enc): all hidden states for each step from encoder
        # padding_mask: (batch_size, seq_len) True if padded token
        s_curr, h_enc, padding_mask = self.encoder(src_batch, src_len)

        # Init first input for target sentence as <sos>-idx
        # (batch_size)
        y_bef = torch.full(
            size=(src_batch.shape[0], 1), fill_value=self.init_token_idx
        ).to(self.device)

        # Safe sequence length of target seqs
        seq_len_trg = trg_batch.shape[1]

        # Init output tensor
        out_dec_all = torch.zeros(
            size=(trg_batch.shape[0], trg_batch.shape[1], self.decoder.vocab_size_trg)
        )
        out_dec_all = out_dec_all.to(self.device)

        for step in range(1, seq_len_trg):
            # Compute attention weights
            # (batch_size, padded_seq_len)
            attention_weights = self.attention(
                hidden_dec=s_curr, hidden_enc=h_enc, padding_mask=padding_mask
            )

            # Compute c_i - enc hidden state summary based on attention weights
            # (batch_size, 2*hidden_dim_enc)
            c_i = weighted_sum(H=h_enc, W=attention_weights.to(self.device))

            # Pass all inputs to decoder to get output and next hidden state
            next_output, s_curr = self.decoder(s_bef=s_curr, y_bef=y_bef, c_i=c_i)

            # Squeeze
            s_curr = s_curr.squeeze()

            # Add dimension if s_curr is just one batch - error fix
            if len(s_curr.shape) == 1:
                s_curr = s_curr.unsqueeze(0)

            # Save all outputs for this step
            out_dec_all[:, step, :] = next_output

            # Decide randomly on whether to use teacher force in this step

            # If teacher forcing, set y_bef to true last label
            rand_num = random.uniform(0, 1)
            if rand_num < teacher_forcing:
                y_bef = trg_batch[:, step]
            # Else take predicted one
            else:
                y_bef = torch.argmax(next_output, dim=1)

        return out_dec_all


class Seq2Seq_With_Attention:
    def __init__(
        self,
        lr,
        enc_vocab_size,
        vocab_size_trg,
        enc_emb_dim,
        hidden_dim_enc,
        hidden_dim_dec,
        padding_idx,
        num_layers_enc,
        num_layers_dec,
        emb_dim_trg,
        trg_pad_idx,
        dropout,
        train_attention=True,
        device="cuda",
        seq_beginning_token_idx=2,
    ):
        """
        Wrapper class that holds optimizer, loss function
        and seq2seq model with class method train_step().
        """

        self.device = device

        # Init Encoder/Decoder/Attention
        encoder = Encoder(
            vocab_size=enc_vocab_size,
            emb_dim=enc_emb_dim,
            hidden_dim_enc=hidden_dim_enc,
            hidden_dim_dec=hidden_dim_dec,
            padding_idx=padding_idx,
            num_layers=num_layers_enc,
            dropout=dropout,
        )

        decoder = Decoder(
            hidden_dim_enc=hidden_dim_enc,
            hidden_dim_dec=hidden_dim_dec,
            emb_dim_trg=emb_dim_trg,
            vocab_size_trg=vocab_size_trg,
            num_layers=num_layers_dec,
            dropout=dropout,
        )

        if train_attention:
            attention = Attention(
                hidden_dim_enc=hidden_dim_enc,
                hidden_dim_dec=hidden_dim_dec,
            )

        else:
            attention = UniformAttention()

        # Init the full pipeline
        self.seq2seq = Seq2Seq_Architecture_with_Att(
            encoder=encoder,
            decoder=decoder,
            attention=attention,
            device=device,
            init_token_idx=seq_beginning_token_idx,
        )

        # Init loss - ignore padding idxs in loss/gradient computations
        self.loss_func = torch.nn.CrossEntropyLoss(ignore_index=trg_pad_idx)

        # Init optimizer
        self.optimizer = torch.optim.Adam(self.seq2seq.parameters(), lr=lr)

    def train_step(self, src_batch, trg_batch, src_lens, teacher_forcing):
        """
        Take one gradient step on the training batch.
        """
        src_batch = src_batch.to(self.device)
        trg_batch = trg_batch.to(self.device)

        # Get decoder outputs for each token of the target sentence
        # (batch_size, trg_seq_len, vocab_size_trg)
        decoder_out = self.seq2seq(
            src_batch=src_batch,
            trg_batch=trg_batch,
            src_len=src_lens,
            teacher_forcing=teacher_forcing,
        )

        # Compute loss
        # Swap dim 2 and dim 1 to get
        # input=(N, C, seq_len) and target=(N, seq_len)
        decoder_out = decoder_out.permute(0, 2, 1)

        loss = self.loss_func(decoder_out, trg_batch)

        # Compute gradients
        self.optimizer.zero_grad()
        loss.backward()

        # Take learning step
        self.optimizer.step()

        return loss.item()

    def set_train(self):
        """
        Set model in training mode.
        """
        self.seq2seq.train()

    def set_train(self):
        """
        Set model in eval mode.
        """
        self.seq2seq.eval()

    def send_to_device(self):
        """
        Send model to device.
        """
        self.seq2seq.to(self.device)
