import sys
from transformers import BertTokenizer, BertModel
from bert_score import score


def calc_bert(gf, pf):
    # Example texts
    reference = "This is a reference text example."
    candidate = "This is a candidate text example."
    # BERTScore calculation
    # scorer = BERTScorer(model_type='bert-base-uncased')
    P, R, F1 = score([candidate], [reference], lang="en", verbose=True)
    print(f"BERTScore Precision: {P.mean():.4f}, Recall: {R.mean():.4f}, F1: {F1.mean():.4f}")


if __name__ == '__main__':
    gold_file = "data/parsed/D1_5_word_groups.csv"
    pred_file = "data/withenc/D1_5_word_groups.csv"
    calc_bert(gold_file, pred_file)



