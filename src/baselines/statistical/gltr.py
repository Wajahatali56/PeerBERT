"""
GLTR (Gehrmann et al., ACL 2019 demo). Features: fractions of tokens whose 1-indexed rank
under the scoring LM is in top-10 / 11-100 / 101-1000 / >1000. A logistic regression on
these 4 features is fit on a whole-paper sample of the TRAIN split only (row cap disclosed
in baselines.yaml common.train_rows). Output = P(AI). Trained (feature-based) detector.
"""

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

FEATURES = ["gltr_top10", "gltr_10_100", "gltr_100_1000", "gltr_over_1000"]


def fit(stats_train, y_train, seed=42):
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, random_state=seed))
    return clf.fit(stats_train[FEATURES].to_numpy(dtype=float), y_train)


def score(clf, stats):
    return clf.predict_proba(stats[FEATURES].to_numpy(dtype=float))[:, 1]
