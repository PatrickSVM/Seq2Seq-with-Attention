import wandb
from seq2seq_attention.train import train_seq2seq_with_attention

if __name__ == "__main__":

    EXP_NAME = "Experiment_7"

    LR = 5e-4
    BATCH_SIZE = 256
    EPOCHS = 25
    MAX_VOCAB_SIZE = 8000
    MIN_FREQ = 2
    ENC_EMB_DIM = 256
    HIDDEN_DIM_ENC = 512
    HIDDEN_DIM_DEC = 512
    NUM_LAYERS_ENC = 1
    NUM_LAYERS_DEC = 1
    EMB_DIM_TRG = 256
    DEVICE = "cuda"
    TEACHER_FORCING = 0.7
    TRAIN_DIR = "./data/processed/train.csv"
    # VAL_DIR = TRAIN_DIR
    # TEST_DIR = TRAIN_DIR
    VAL_DIR = "./data/processed/val.csv"
    TEST_DIR = "./data/processed/val.csv"
    PROGRESS_BAR = False
    USE_WANDB = True

    # Setup hyperparams for wandb
    hyper_params = {
        "lr": LR,
        "batch_size": BATCH_SIZE,
        "epochs": EPOCHS,
        "max_vocab_size": MAX_VOCAB_SIZE,
        "min_freq": MIN_FREQ,
        "enc_hidden": HIDDEN_DIM_ENC,
        "dec_hidden": HIDDEN_DIM_DEC,
        "embedding_enc": ENC_EMB_DIM,
        "embedding_dec": ENC_EMB_DIM,
    }

    # Init wandb
    if USE_WANDB:
        wandb.init(
            project="Seq2Seq-With-Attention",
            name=EXP_NAME,
            # track hyperparameters and run metadata
            config=hyper_params,
        )

    train_seq2seq_with_attention(
        lr=LR,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        enc_emb_dim=ENC_EMB_DIM,
        hidden_dim_enc=HIDDEN_DIM_ENC,
        hidden_dim_dec=HIDDEN_DIM_DEC,
        num_layers_enc=NUM_LAYERS_ENC,
        num_layers_dec=NUM_LAYERS_DEC,
        emb_dim_trg=EMB_DIM_TRG,
        max_vocab_size=MAX_VOCAB_SIZE,
        min_freq=MIN_FREQ,
        device=DEVICE,
        teacher_forcing=TEACHER_FORCING,
        train_dir=TRAIN_DIR,
        val_dir=VAL_DIR,
        test_dir=TEST_DIR,
        progress_bar=PROGRESS_BAR,
        use_wandb=USE_WANDB,
        exp_name=EXP_NAME,
    )
