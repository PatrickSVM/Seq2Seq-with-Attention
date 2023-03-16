import numpy as np
import wandb
from tqdm import tqdm
from seq2seq_attention.evaluate import evaluate
from seq2seq_attention.model import Seq2Seq_With_Attention
from seq2seq_attention.build_dataloaders import (
    build_fields,
    build_bucket_iterator,
    get_datasets,
    build_vocab,
)


def train_seq2seq_with_attention(
    lr,
    batch_size,
    epochs,
    enc_emb_dim,
    hidden_dim_enc,
    hidden_dim_dec,
    num_layers_enc,
    num_layers_dec,
    emb_dim_trg,
    device,
    teacher_forcing,
    train_dir,
    val_dir,
    test_dir,
    progress_bar=False,
    use_wandb=False,
):

    """
    Wrapper function that trains a seq2seq model with Bahdanau
    attention given the provided parameters.
    """

    disable_pro_bar = not progress_bar

    # Build fields for german and english
    src_field, trg_field = build_fields()

    # Get datasets
    train_set, val_set, test_set = get_datasets(
        train_path=train_dir,
        val_path=val_dir,
        test_path=test_dir,
        src_field=src_field,
        trg_field=trg_field,
    )

    # Build vocabularies
    build_vocab(src_field=src_field, trg_field=trg_field, train_set=train_set)

    # Get data loaders
    train_loader = build_bucket_iterator(
        dataset=train_set, batch_size=batch_size, device=device
    )
    val_loader = build_bucket_iterator(
        dataset=val_set, batch_size=batch_size, device=device
    )

    # Load test set - if val_dir=test_dir, set it to val_loader
    real_test = val_dir != test_dir
    if not real_test:
        test_loader = val_loader
    else:
        # Test model on real test set
        test_loader = build_bucket_iterator(
            dataset=test_set, batch_size=device, device=device
        )

    # Get padding/<sos> idxs
    src_pad_idx = src_field.vocab.stoi["<pad>"]
    trg_pad_idx = trg_field.vocab.stoi["<pad>"]
    seq_beginning_token_idx = src_field.vocab.stoi["<sos>"]
    assert src_field.vocab.stoi["<sos>"] == trg_field.vocab.stoi["<sos>"]

    # Init model wrapper class
    model = Seq2Seq_With_Attention(
        lr=lr,
        enc_vocab_size=len(src_field.vocab),
        vocab_size_trg=len(trg_field.vocab),
        enc_emb_dim=enc_emb_dim,
        hidden_dim_enc=hidden_dim_enc,
        hidden_dim_dec=hidden_dim_dec,
        padding_idx=src_pad_idx,
        num_layers_enc=num_layers_enc,
        num_layers_dec=num_layers_dec,
        emb_dim_trg=emb_dim_trg,
        trg_pad_idx=trg_pad_idx,
        device=device,
        seq_beginning_token_idx=seq_beginning_token_idx,
    )

    # Send model to device
    model.send_to_device()

    train_losses = np.zeros(epochs)
    val_losses = np.zeros(epochs)

    for epoch in range(epochs):

        # Init loss stats for epoch
        epoch_loss = 0

        for n_batch, train_batch in enumerate(
            tqdm(
                train_loader,
                desc=f"Epoch {epoch}",
                unit="batch",
                disable=disable_pro_bar,
            )
        ):
        
            model.seq2seq.train()

            # Take one gradient step
            epoch_loss += model.train_step(
                src_batch=train_batch.src[0],
                trg_batch=train_batch.trg,
                src_lens=train_batch.src[1],
                teacher_forcing=teacher_forcing,
            )

        # Evaluate
        eval_loss = evaluate(model=model, eval_loader=val_loader)

        # Save mean train/val loss
        train_losses[epoch] = epoch_loss / len(train_loader)
        val_losses[epoch] = eval_loss

        print(
            f"Epoch {epoch}: Train loss [{train_losses[epoch]}]   |  Val loss [{eval_loss}]"
        )

        # Logging
        if use_wandb:
            epoch_log_res = {"Train loss": train_losses[epoch], "Val loss": eval_loss}

            wandb.log(epoch_log_res)
