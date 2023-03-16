import torch


def evaluate(model, eval_loader):

    # Set in evaluation mode
    model.seq2seq.eval()

    eval_loss = 0

    with torch.no_grad():

        for i, batch in enumerate(eval_loader):

            src, src_len = batch.src
            trg = batch.trg

            # Get decoder outputs for each token of the target sentence
            # (batch_size, trg_seq_len, vocab_size_trg)
            decoder_out = model.seq2seq(
                src_batch=src,
                trg_batch=trg,
                src_len=src_len,
                teacher_forcing=0,
            )

            # Compute loss
            # Swap dim 2 and dim 1 to get
            # input=(N, C, seq_len) and target=(N, seq_len)
            decoder_out = decoder_out.permute(0, 2, 1)
            loss = model.loss_func(decoder_out, trg)

            eval_loss += loss.item()

    # Get average loss per sentence pair
    mean_eval_loss = eval_loss / len(eval_loader)

    return mean_eval_loss
