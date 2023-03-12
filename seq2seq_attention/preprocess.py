import pandas as pd
import csv


def get_parallel_csv(path_1, path_2, new_file_path, delimiter):
    """
    Take two txts with parallel translations, create csv
    file with two cols for src and trg.
    """
    removed = 0
    with open(path_1) as src, open(path_2) as tgt:
        with open(new_file_path, "w") as file:
            for src_sentence, tgt_sentence in zip(src, tgt):
                if src_sentence.isspace() or tgt_sentence.isspace():
                    removed += 1
                    continue
                line = f"{src_sentence.rstrip()} {delimiter} {tgt_sentence.rstrip()}"
                file.write(line)
                file.write("\n")

    print("File successfully created.")
    print(f"{removed} lines were removed.")


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
