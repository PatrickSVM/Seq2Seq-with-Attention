from functools import partial
from sacrebleu.metrics import BLEU

def get_bleu_sentence(targ_raw, trans, scorer_class, trg_field, ):
    """ 
    Get bleu score for a sentence, given a translation yielded
    by the translation method.
    """
    
    # Get reference
    trg_tokenized = trg_field.tokenize(targ_raw)
    reference = " ".join([tok.lower() for tok in trg_tokenized])

    score = scorer_class.sentence_score(hypothesis=trans, references=[reference]).score

    return score


def get_bleu_dataset(dataset, trg_field, trans_col, parallel=False):
    bleu_params = dict(effective_order=True, tokenize=None, smooth_method="floor", smooth_value=0.01)
    bleu = BLEU(**bleu_params)

    sentence_bleu = partial(get_bleu_sentence, 
                  scorer_class=bleu,  
                  trg_field=trg_field,)
    
    if parallel:
        bleu_col = dataset.apply(lambda row: sentence_bleu(targ_raw=row.eng, trans=row[trans_col]), axis=1)
        res = bleu_col.mean()
    
    else:
        cum_bleu = 0
        for row in dataset.iterrows():
            row = row[1]
            cum_bleu += sentence_bleu(targ_raw=row.eng, trans=row[trans_col])

        res = cum_bleu/len(dataset)

    return res


def get_bleu_col(dataset, trg_field, trans_col):
    bleu_params = dict(effective_order=True, tokenize=None, smooth_method="floor", smooth_value=0.01)
    bleu = BLEU(**bleu_params)

    sentence_bleu = partial(get_bleu_sentence, 
                  scorer_class=bleu,  
                  trg_field=trg_field,)
    
    bleu_col = dataset.apply(lambda row: sentence_bleu(targ_raw=row.eng, trans=row[trans_col]), axis=1)

    return bleu_col