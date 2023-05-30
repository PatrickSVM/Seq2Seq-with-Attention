import random
import numpy as np
import wandb
import torch
from tqdm import tqdm
from time import time
from seq2seq_attention.evaluate import evaluate
from seq2seq_attention.model import Seq2Seq_With_Attention
from seq2seq_attention.build_dataloaders import (
    build_fields,
    build_bucket_iterator,
    get_datasets,
    build_vocab,
)
from seq2seq_attention.translate import translate_sentence
from seq2seq_attention.model_saver import SaveBestModel


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
    dropout,
    device,
    teacher_forcing,
    max_vocab_size,
    min_freq,
    train_dir,
    val_dir,
    test_dir,
    train_attention=True,
    progress_bar=False,
    use_wandb=False,
    exp_name="",
):

    """
    Wrapper function that trains a seq2seq model with Bahdanau
    attention given the provided parameters.
    """

    # Set seeds
    random.seed(118)
    torch.manual_seed(999)

    disable_pro_bar = not progress_bar

    # Specify some examples
    examples = [
        "Das Land hat Schulden.",
        "Ein Mann fährt mit dem Auto.",
        "Vielen Dank, dass sie mir helfen.",
        "Wissen steht in Büchern.",
        "Nur Kriminelle können diese mit Leichtigkeit erfüllen, aber Studenten, Forscher und Journalisten nicht.",
        "Vielen Dank, Herr Fischler, für Ihr Engagement in der Fragestunde heute Nachmittag.",
        "Wir müssen endlich etwas unternehmen.",
        "Leider war das nicht der Fall.",
        "Ich frage die Kommission, sind Sie von den Niederlanden darüber unterrichtet worden?",
        "Ich glaube, auf manche Angriffe muss man nicht antworten.",
        "Alle Probleme sollten vom Volk selbst gelöst werden, und zwar auf seine Weise.",
    ]

    # Init model saver
    model_saver = SaveBestModel(out_dir=f"{exp_name}")

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
    build_vocab(
        src_field=src_field,
        trg_field=trg_field,
        train_set=train_set,
        max_vocab_size=max_vocab_size,
        min_freq=min_freq,
    )

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

    # Safe number of batches in train loader and eval points
    perc = 0.25
    n_batches_train = len(train_loader)
    eval_points = [
        round(i * perc * n_batches_train) - 1 for i in range(1, round(1 / perc))
    ]
    eval_points.append(n_batches_train - 1)

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
        dropout=dropout,
        padding_idx=src_pad_idx,
        num_layers_enc=num_layers_enc,
        num_layers_dec=num_layers_dec,
        emb_dim_trg=emb_dim_trg,
        trg_pad_idx=trg_pad_idx,
        device=device,
        seq_beginning_token_idx=seq_beginning_token_idx,
        train_attention=train_attention,
    )

    # Send model to device
    model.send_to_device()

    train_losses = []
    val_losses = []

    for epoch in range(epochs):

        now = time()

        # Init loss stats for epoch
        train_loss = 0

        n_batches_since_eval = 0

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
            train_loss += model.train_step(
                src_batch=train_batch.src[0],
                trg_batch=train_batch.trg,
                src_lens=train_batch.src[1],
                teacher_forcing=teacher_forcing,
            )

            n_batches_since_eval += 1

            # Calculate and safe train/eval losses at 25% of epoch
            if n_batch in eval_points:

                # Translate example
                for i, ex in enumerate(examples):
                    translation, _, _, _ = translate_sentence(
                        sentence=ex,
                        seq2seq_model=model.seq2seq,
                        src_field=src_field,
                        bos=src_field.init_token,
                        eos=src_field.eos_token,
                        eos_idx=src_field.vocab.stoi[src_field.eos_token],
                        trg_field=trg_field,
                        max_len=30,
                    )
                    print(f"Example #{i}:")
                    print(examples[i], " - ", translation, "\n")

                now_eval = time()

                # Evaluate
                eval_loss = evaluate(model=model, eval_loader=val_loader)

                print(f"Evaluation time: {(time()-now_eval)/60:.2f} minutes.")

                # Save mean train/val loss
                train_losses.append(train_loss / n_batches_since_eval)
                val_losses.append(eval_loss)

                # Set counter to 0 again
                n_batches_since_eval = 0
                train_loss = 0

                print(
                    f"Epoch {epoch} [{round(n_batch*100/n_batches_train)}%]: Train loss [{train_losses[-1]}]   |  Val loss [{eval_loss}]\n"
                )
                print("##########################################\n")

                # Logging
                if use_wandb:
                    epoch_log_res = {
                        "Train loss": train_losses[-1],
                        "Val loss": eval_loss,
                    }

                    wandb.log(epoch_log_res)

                # Check for best model
                model_saver(val_loss=eval_loss, epoch=epoch, model=model.seq2seq)

        print(f"Epoch Training time: {(time()-now)/60:.2f} minutes.")
