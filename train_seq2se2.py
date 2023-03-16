from seq2seq_attention.train import train_seq2seq_with_attention

if __name__ == "__main__":

    LR = 1e-3
    BATCH_SIZE = 80
    EPOCHS = 10
    MAX_VOCAB_SIZE = 32000
    ENC_EMB_DIM = 310
    HIDDEN_DIM_ENC = 500
    HIDDEN_DIM_DEC = 500
    NUM_LAYERS_ENC = 1
    NUM_LAYERS_DEC = 1
    EMB_DIM_TRG = 310
    DEVICE = "cuda"
    TEACHER_FORCING = True
    TRAIN_DIR = "./data/processed/train_mini.csv"
    VAL_DIR = TRAIN_DIR
    TEST_DIR = TRAIN_DIR
    #VAL_DIR = "./data/processed/val.csv"
    #TEST_DIR = "./data/processed/val.csv"
    PROGRESS_BAR = True

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
        device=DEVICE,
        teacher_forcing=TEACHER_FORCING,
        train_dir=TRAIN_DIR,
        val_dir=VAL_DIR,
        test_dir=TEST_DIR,
        progress_bar=PROGRESS_BAR,
    )
