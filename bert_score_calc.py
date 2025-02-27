import sys
from transformers import BertTokenizer, BertModel
from bert_score import score


class ScoreObj:
    def __init__(self, P, R, F1):
        self.P = P
        self.R = R
        self.F1 = F1

def calc_bert(g_lines, p_lines):
    # Example texts
    reference = "This is a reference text example."
    candidate = "This is a candidate text example."
    scores = []
    # BERTScore calculation
    # scorer = BERTScorer(model_type='bert-base-uncased')
    for g, p in zip(g_lines, p_lines):
        P, R, F1 = score([g], [p], lang="en", verbose=True)
        scores.append(ScoreObj(P, R, F1))


    for s in scores:
        print(f"BERTScore Precision: {s.P.mean():.4f}, Recall: {s.R.mean():.4f}, F1: {s.F1.mean():.4f}")


if __name__ == '__main__':
    gold_file = "data/parsed/D1_5_word_groups.txt"
    pred_file = "data/encoded/D1_5_word_groups_2.txt"

    with open(gold_file, 'r') as file:
        gold_lines = file.readlines()
        # Remove trailing newline characters from each line
        gold_lines = [line.rstrip('\n') for line in gold_lines]
        gold_lines = [string for string in gold_lines if string]

    with open(pred_file, 'r') as file:
        pred_lines = file.readlines()
        # Remove trailing newline characters from each line
        pred_lines = [line.rstrip('\n') for line in pred_lines]
        pred_lines = [string for string in pred_lines if string]

    calc_bert(gold_lines, pred_lines)



