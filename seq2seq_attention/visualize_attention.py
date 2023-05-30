import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from seq2seq_attention.translate import translate_sentence


def get_attention_frames(sentences, model, src_field, trg_field):
    # Translate examples
    attention_frames = []
    translations = []
    for i, sent in enumerate(sentences):
        translation, attention, _, _ = translate_sentence(
            sentence=sent,
            seq2seq_model=model.seq2seq,
            src_field=src_field,
            bos=src_field.init_token,
            eos=src_field.eos_token,
            eos_idx=src_field.vocab.stoi[src_field.eos_token],
            trg_field=trg_field,
            max_len=30,
        )

        # Cut away bos in attention since bos created
        attention = attention[:, 1:]

        # To pandas
        frame = pd.DataFrame(attention)
        frame.columns = src_field.tokenize(sent) + ["<eos>"]
        frame.index = trg_field.tokenize(translation.replace("<unk>", "UNK")) + [
            "<eos>"
        ]

        # Save in list
        attention_frames.append(frame.transpose())
        translations.append(translation)

    return translations, attention_frames


def plot_attention(attention_frame):
    sns.heatmap(attention_frame, cmap="bone", vmin=0, vmax=1)
    plt.yticks(rotation=0)
    plt.gca().xaxis.tick_top()
    plt.gca().xaxis.set_label_position("top")
    plt.xticks(rotation=45)

    return plt.gcf()
