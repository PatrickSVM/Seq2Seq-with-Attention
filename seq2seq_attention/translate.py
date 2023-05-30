import torch
import numpy as np
from seq2seq_attention.model import weighted_sum


def forward_translation_encoder(seq2seq_model, src_batch, src_len):
    # Compute entire forward pass on src batch for all steps
    # h_dec (batch_size, hidden_dim_dec): first hidden state for decoder
    # h_enc (batch_size, seq_len, hidden_dim_enc): all hidden states for each step from encoder
    # padding_mask: (batch_size, seq_len) True if padded token
    s_curr, h_enc, padding_mask = seq2seq_model.encoder(src_batch, src_len)

    return s_curr, h_enc, padding_mask


@torch.no_grad()
def translate_sentence(
    sentence, seq2seq_model, src_field, bos, eos, eos_idx, trg_field, max_len
):
    """
    Function that takes a sentence and translates it.
    """

    seq2seq_model.eval()

    # Tokenize sentence
    src_tokenized = src_field.tokenize(sentence)

    # Add bos/eos tokens
    src_tokenized = [bos] + src_tokenized + [eos]

    # Convert to indexes
    src_numerical = [src_field.vocab.stoi[token.lower()] for token in src_tokenized]

    # Convert to tensor (1, num_tokens)
    src_tensor = torch.LongTensor(src_numerical).unsqueeze(0)

    # Get encoder sentence summary s_curr,  all hidden states h_enc and padding mask
    s_curr, h_enc, padding_mask = seq2seq_model.encoder(
        src_tensor.to(seq2seq_model.device),
        torch.LongTensor([len(src_numerical)]).to(seq2seq_model.device),
    )

    # Init first input for target sentence as <sos>-idx
    # (batch_size)
    y_bef = torch.full(size=(1, 1), fill_value=seq2seq_model.init_token_idx).to(
        seq2seq_model.device
    )

    # Init translation and attention weights
    translation = []
    attention_weights_all = []

    for step in range(1, max_len):
        # Compute attention weights
        # (batch_size, padded_seq_len)
        attention_weights = seq2seq_model.attention(
            hidden_dec=s_curr, hidden_enc=h_enc, padding_mask=padding_mask
        )

        # Compute c_i - enc hidden state summary based on attention weights
        # (batch_size, 2*hidden_dim_enc)
        c_i = weighted_sum(H=h_enc, W=attention_weights.to(seq2seq_model.device))

        # Pass all inputs to decoder to get output and next hidden state
        next_output, s_curr = seq2seq_model.decoder(s_bef=s_curr, y_bef=y_bef, c_i=c_i)

        # Squeeze
        s_curr = s_curr.squeeze().unsqueeze(0)

        # Get max prediction as next word
        y_bef = torch.argmax(next_output, dim=1)

        # Check if EOS token
        if y_bef == eos_idx:
            attention_weights_all.append(attention_weights.squeeze().cpu().tolist())
            break

        # Add to translation
        translation.append(y_bef)

        # Save attention
        attention_weights_all.append(attention_weights.squeeze().cpu().tolist())

    # Get sentence as words
    translation_tok = [trg_field.vocab.itos[idx] for idx in translation]
    translation = " ".join(translation_tok)

    return translation, np.array(attention_weights_all), src_tokenized, translation_tok




@torch.no_grad()
def translate_sentence_without(
    sentence, seq2seq_model, src_field, bos, eos, eos_idx, trg_field, max_len
):
    """
    Function that takes a sentence and translates it.
    """

    seq2seq_model.eval()

    # Tokenize sentence
    src_tokenized = src_field.tokenize(sentence)

    # Add bos/eos tokens
    src_tokenized = [bos] + src_tokenized + [eos]

    # Convert to indexes
    src_numerical = [src_field.vocab.stoi[token.lower()] for token in src_tokenized]

    # Convert to tensor (1, num_tokens)
    src_tensor = torch.LongTensor(src_numerical).unsqueeze(0)

    # Get encoder sentence summary s_curr,  all hidden states h_enc and padding mask
    s_curr, h_enc, padding_mask = seq2seq_model.encoder(
        src_tensor.to(seq2seq_model.device),
        torch.LongTensor([len(src_numerical)]).to(seq2seq_model.device),
    )

    # Init first input for target sentence as <sos>-idx
    # (batch_size)
    y_bef = torch.full(size=(1, 1), fill_value=seq2seq_model.init_token_idx).to(
        seq2seq_model.device
    )

    # Init translation and attention weights
    translation = []

    for step in range(1, max_len):
        # Compute attention weights
        # (batch_size, padded_seq_len)
        attention_weights = seq2seq_model.attention(
            hidden_dec=s_curr, hidden_enc=h_enc, padding_mask=padding_mask
        )

        # Compute c_i - enc hidden state summary based on attention weights
        # (batch_size, 2*hidden_dim_enc)
        c_i = weighted_sum(H=h_enc, W=attention_weights.to(seq2seq_model.device))

        # Pass all inputs to decoder to get output and next hidden state
        next_output, s_curr = seq2seq_model.decoder(s_bef=s_curr, y_bef=y_bef, c_i=c_i)

        # Squeeze
        s_curr = s_curr.squeeze().unsqueeze(0)

        # Get max prediction as next word
        y_bef = torch.argmax(next_output, dim=1)

        # Check if EOS token
        if y_bef == eos_idx:
            break

        # Add to translation
        translation.append(y_bef)

    # Get sentence as words
    translation_tok = [trg_field.vocab.itos[idx] for idx in translation]
    translation = " ".join(translation_tok)

    return translation