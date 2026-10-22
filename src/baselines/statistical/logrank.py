"""
LogRank (Log-Rank baseline; Ippolito et al., ACL 2020 rank-based detection; as implemented in
the DetectGPT/DetectLLM baseline code): score = - mean log-rank of the observed tokens
under a scoring LM (1-indexed ranks). Larger = more likely machine-generated. Zero-shot.
"""


def score(stats):
    return -stats["mean_logrank"].to_numpy(dtype=float)
