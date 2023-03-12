import spacy
from torchtext.data import TabularDataset
from torchtext.data import Field, BucketIterator


spacy_de = spacy.load("de_core_news_sm")
spacy_en = spacy.load("en_core_web_sm")


def tokenize_ger(text):
    """
    Take german sentence and tokenize it using spacy.
    """
    return [tok.text for tok in spacy_de.tokenizer(text)]


def tokenize_eng(text):
    """
    Take english sentence and tokenize it using spacy.
    """
    return [tok.text for tok in spacy_en.tokenizer(text)]


def build_fields(tokenizer_src=tokenize_ger, tokenizer_trg=tokenize_eng):
    """
    Build fields for source and target sentences.

    End-Padding for packed_sequence usage,
    Lengths are included for packed_sequence usage.
    """
    src_field = Field(
        tokenize=tokenizer_src,
        init_token="<sos>",
        eos_token="<eos>",
        pad_token="<pad>",
        unk_token="<unk>",
        lower=True,
        include_lengths=True,
        sequential=True,
        batch_first=True,
        use_vocab=True,
        pad_first=False,
    )

    trg_field = Field(
        tokenize=tokenizer_trg,
        init_token="<sos>",
        eos_token="<eos>",
        pad_token="<pad>",
        unk_token="<unk>",
        lower=True,
        include_lengths=True,
        sequential=True,
        batch_first=True,
        use_vocab=True,
        pad_first=False,
    )

    return src_field, trg_field


def get_datasets(train_path, val_path, test_path, src_field, trg_field):
    """
    Load created csv files as TabularDatasets.

    Note: The specific delimiter is provided in the function.
    """
    train_set, valid_set, test_set = TabularDataset.splits(
        path="",
        train=train_path,
        validation=val_path,
        test=test_path,
        format="csv",
        csv_reader_params={"delimiter": ">", "skipinitialspace":True},
        fields=[("src", src_field), ("trg", trg_field)],
    )

    return train_set, valid_set, test_set


def build_vocab(src_field, trg_field, train_set, min_freq=2, max_vocab_size=32000):
    """
    Build vocabs with specified vocab size and min frequency of words
    based on training set.
    """
    src_field.build_vocab(train_set, min_freq=min_freq, max_size=max_vocab_size)
    trg_field.build_vocab(train_set, min_freq=min_freq, max_size=max_vocab_size)


def build_bucket_iterator(dataset, batch_size, device):
    """
    Build BucketIterator for dataset.

    Sorts seqs based on seq size to minimize padding
    and improve efficiency. Batches get shuffled each
    new epoch.

    Check https://torchtext.readthedocs.io/en/latest/data.html#bucketiterator.
    """
    iterator = BucketIterator(
        dataset=dataset,
        batch_size=batch_size,
        sort_key=lambda x: len(x.src),
        shuffle=True,
        device=device,
        sort=True,
    )

    return iterator
