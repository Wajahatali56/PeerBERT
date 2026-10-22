"""
DetectLLM-LRR (Su et al., 2023, "DetectLLM: Leveraging Log Rank Information for Zero-Shot
Detection of Machine-Generated Text"). Official code (mbzuai-nlp/DetectLLM,
baselines/all_baselines.py): LRR = -mean_ll / mean_logrank, i.e. mean NLL / mean log-rank.
Larger = more likely machine-generated. Zero-shot.

Deviation (disclosed): epsilon added to the denominator, because mean log-rank is 0 when
every token is the model's top-1 prediction.
"""


EPS = 1e-6


def score(stats, eps=EPS):
    nll = -stats["mean_ll"].to_numpy(dtype=float)
    return nll / (stats["mean_logrank"].to_numpy(dtype=float) + eps)
