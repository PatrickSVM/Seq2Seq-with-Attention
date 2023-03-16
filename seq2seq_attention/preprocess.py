import pandas as pd
import csv
from seq2seq_attention.build_dataloaders import tokenize_ger, tokenize_eng


def get_parallel_csv(path_1, path_2, new_file_path, delimiter):
    """
    Take two txts with parallel translations, create csv
    file with two cols for src and trg.

    The function also replaces the delimiter char in every of
    the sentences.
    """
    removed = 0
    with open(path_1) as src, open(path_2) as tgt:
        with open(new_file_path, "w") as file:
            for src_sentence, tgt_sentence in zip(src, tgt):
                if src_sentence.isspace() or tgt_sentence.isspace():
                    removed += 1
                    continue
                line = f"{src_sentence.replace(delimiter, '').rstrip()} {delimiter} {tgt_sentence.replace(delimiter, '').rstrip()}"
                file.write(line)
                file.write("\n")

    print("File successfully created.")
    print(f"{removed} lines were removed.")


def remove_sentences(
    data_dir,
    min_length,
    max_length,
    new_file_path,
    delimiter,
    tokenizer_src=tokenize_ger,
    tokenizer_trg=tokenize_eng,
):
    """
    Takes parallel csv file and removes sentence-pairs in the source
    sentence and target_sentence, that are shorter/longer than min/max_length.
    """
    removed = 0
    with open(data_dir, "r") as file:
        with open(new_file_path, "w") as new_file:
            for line in file:
                src = line.split(delimiter)[0]
                if len(tokenizer_src(src)) < min_length:
                    removed += 1
                    continue
                if len(tokenizer_src(src)) > max_length:
                    removed += 1
                    continue
                trg = line.split(delimiter)[1]
                if len(tokenizer_trg(trg)) < min_length:
                    removed += 1
                    continue
                if len(tokenizer_trg(trg)) > max_length:
                    removed += 1
                    continue
                new_file.write(line)
                new_file.write("\n")
    print("File successfully created.")
    print(f"{removed} sentence-pairs were removed.")


def train_test_split(file_path, sep, dir, random_seed=118):
    """
    Read full training set, split it in train, val, test set
    with ratio (0.8, 0.1, 0.1).
    """
    # Set seed for numpy (pandas falls back there)
    full_set = pd.read_csv(
        file_path, sep=sep, on_bad_lines="warn", quoting=csv.QUOTE_NONE, header=None
    )
    num_train, num_val = round(0.8 * len(full_set)), round(0.1 * len(full_set))
    num_test = len(full_set) - num_train - num_val

    # Shuffle dataset
    full_set = full_set.sample(frac=1, random_state=random_seed)

    # Infer all three sets
    train = full_set.iloc[:num_train, :]
    val = full_set.iloc[num_train : num_train + num_val, :]
    test = full_set.iloc[-num_test:, :]

    train.to_csv(f"{dir}/train.csv", sep=sep, header=None, index=False)
    val.to_csv(f"{dir}/val.csv", sep=sep, header=None, index=False)
    test.to_csv(f"{dir}/test.csv", sep=sep, header=None, index=False)

    print("All files successfully created.")
