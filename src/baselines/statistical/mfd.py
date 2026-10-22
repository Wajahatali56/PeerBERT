"""
MFD - multi-feature detector (Xiang & Wu, 2023, Research Square preprint; NOT peer reviewed;
no official code found). Features from one scoring-LM pass:
  mean log-likelihood, mean log-rank, mean entropy, LLM-Deviation = mean (log rank)^2.
Head (fit on a whole-paper TRAIN sample only):
  logreg  "MFD"   - the variant named MFD in the preprint's ablation (primary)
  mlp     "MFD-M" - the 4-16-16-16-1 network described in the preprint's text
The preprint is internally inconsistent about which head is "MFD"; both are implemented
and the choice is disclosed. Features are standardised before either head.
"""

from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

FEATURES = ["mean_ll", "mean_logrank", "mean_entropy", "mean_logrank_sq"]


def fit(stats_train, y_train, head="logreg", hidden=(16, 16, 16), seed=42):
    if head == "logreg":
        model = LogisticRegression(max_iter=2000, random_state=seed)
    elif head == "mlp":
        model = MLPClassifier(hidden_layer_sizes=tuple(hidden), max_iter=2000, early_stopping=True,
                              random_state=seed)
    else:
        raise ValueError(f"unknown MFD head '{head}'")
    return make_pipeline(StandardScaler(), model).fit(stats_train[FEATURES].to_numpy(dtype=float), y_train)


def score(clf, stats):
    return clf.predict_proba(stats[FEATURES].to_numpy(dtype=float))[:, 1]
