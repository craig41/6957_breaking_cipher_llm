import itertools
import os
import re
import argparse

import pandas as pd
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

    p = []
    r = []
    f1 = []
    for s in scores:
        p.append(s.P.mean())
        r.append(s.R.mean())
        f1.append(s.F1.mean())
        # print(f"BERTScore Precision: {s.P.mean():.4f}, Recall: {s.R.mean():.4f}, F1: {s.F1.mean():.4f}")
    
    return scores, p, r, f1

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-g",
        "--gold",
        type=str,
        help="gold standard file to compare against"
    )
    parser.add_argument(
        "-t",
        "--translations",
        type=str,
        default=None,
        help="translation file to compare against"
    )
    parser.add_argument(
        "-o",
        "--output",
         type=str,
         default=None
    )
    args = parser.parse_args()

    with open(args.gold, "r", encoding="utf-8") as f:
        gold_lines = f.readlines()
    with open(args.translations, "r", encoding="utf-8") as f:
        pred_lines = f.readlines()

    scores, p, r, f1 = calc_bert(gold_lines, pred_lines)
    
    score_df = pd.DataFrame()
    score_df["P"] = p
    score_df["R"] = r
    score_df["F1"] = f1

    score_txt = []
    score_txt.append("Mean Precision for {}: {:.4f}".format(args.translations, score_df['P'].mean()))
    score_txt.append("Max Precision for {}: {:.4f}".format(args.translations, score_df['P'].max()))
    score_txt.append("Min Precision for {}: {:.4f}".format(args.translations, score_df['P'].min()))
    score_txt.append("Mean Recall for {}: {:.4f}".format(args.translations, score_df['R'].mean()))
    score_txt.append("Max Recall for {}: {:.4f}".format(args.translations, score_df['R'].max()))
    score_txt.append("Min Recall for {}: {:.4f}".format(args.translations, score_df['R'].min()))
    score_txt.append("Mean F1 for {}: {:.4f}".format(args.translations, score_df['F1'].mean()))
    score_txt.append("Max F1 for {}: {:.4f}".format(args.translations, score_df['F1'].max()))
    score_txt.append("Min F1 for {}: {:.4f}".format(args.translations, score_df['F1'].min()))

    with open(args.output, "w") as txt_file:
        for line in score_txt:
            txt_file.write(line + "\n")