"""
Registry of the 8 baselines: paper, code source, detector type, what was verified in this
project, and disclosed deviations. Types match the comparison tables:
  zero-shot detector   - no training on our data (threshold calibrated on validation)
  pretrained detector  - released detector checkpoint, not trained on our data
  trained classifier   - fit on our TRAIN split (here: small feature-based heads)
All detectors return scores where LARGER = MORE LIKELY AI.
"""

BASELINES = {
    "logrank": {
        "display": "LogRank",
        "type": "zero-shot detector",
        "paper": "Ippolito et al., ACL 2020 (rank-based detection); Log-Rank as in DetectGPT/DetectLLM baselines",
        "code": "mbzuai-nlp/DetectLLM baselines/rank.py (rank 1-indexed, log, mean)",
        "score": "-mean log-rank",
        "verified": ["1-indexed ranks, log, mean over tokens (DetectLLM rank.py, fetched)"],
        "deviations": ["scoring LM = token_stats.model (disclosed)"],
        "needs": "token_stats",
    },
    "gltr": {
        "display": "GLTR",
        "type": "trained classifier",
        "paper": "Gehrmann et al., ACL 2019 (demo)",
        "code": "this repo (rank buckets) + scikit-learn logistic regression",
        "score": "P(AI) from logistic regression on 4 rank-bucket fractions",
        "verified": ["bucket edges 10/100/1000 as in the GLTR visualisation"],
        "deviations": ["LR fit on a capped whole-paper TRAIN sample (common.train_rows, disclosed)"],
        "needs": "token_stats",
    },
    "detectllm_lrr": {
        "display": "DetectLLM-LRR",
        "type": "zero-shot detector",
        "paper": "Su et al., 2023 (DetectLLM)",
        "code": "mbzuai-nlp/DetectLLM baselines/all_baselines.py: -ll / logrank",
        "score": "mean NLL / mean log-rank",
        "verified": ["ratio and sign from the official code (fetched)"],
        "deviations": ["epsilon 1e-6 added to the denominator (disclosed)"],
        "needs": "token_stats",
    },
    "mfd": {
        "display": "MFD",
        "type": "trained classifier",
        "paper": "Xiang & Wu, 2023, Research Square preprint (not peer reviewed; no official repo found)",
        "code": "this repo",
        "score": "P(AI) from LR (MFD) or 4-16-16-16-1 MLP (MFD-M) on 4 LM features",
        "verified": ["feature definitions as stated in the thesis protocol; preprint not re-read in this session"],
        "deviations": ["preprint internally inconsistent about the head; LR used as primary (disclosed)",
                       "fit on capped TRAIN sample"],
        "needs": "token_stats",
    },
    "fast_detectgpt": {
        "display": "Fast-DetectGPT",
        "type": "zero-shot detector",
        "paper": "Bao et al., ICLR 2024",
        "code": "baoguangsheng/fast-detect-gpt scripts/fast_detect_gpt.py::get_sampling_discrepancy_analytic",
        "score": "(sum ll - sum mean_ref) / sqrt(sum var_ref), analytic",
        "verified": ["analytic formula and model ids (fetched from the official repo)"],
        "deviations": ["reference = scoring = token_stats.model (surrogate, disclosed)"],
        "needs": "token_stats",
    },
    "binoculars": {
        "display": "Binoculars",
        "type": "zero-shot detector",
        "paper": "Hans et al., ICML 2024",
        "code": "ahans30/Binoculars binoculars/detector.py + metrics.py",
        "score": "-(log-ppl_performer / x-ppl(observer, performer))",
        "verified": ["ppl uses PERFORMER logits; x_ppl = CE(observer probs, performer log-probs); "
                     "default pair falcon-7b / falcon-7b-instruct (fetched)"],
        "deviations": ["smaller same-tokenizer pair by default on 16 GB GPUs (flagged); "
                       "published thresholds not reused"],
        "needs": "own_models",
    },
    "radar": {
        "display": "RADAR",
        "type": "pretrained detector",
        "paper": "Hu et al., NeurIPS 2023",
        "code": "HF checkpoint TrustSafeAI/RADAR-Vicuna-7B",
        "score": "P(AI) from the released classifier",
        "verified": [],
        "deviations": ["label order UNVERIFIED; orientation decided on VALIDATION only (disclosed)"],
        "needs": "own_models",
    },
    "dnagpt": {
        "display": "DNA-GPT",
        "type": "zero-shot detector",
        "paper": "Yang et al., ICLR 2024",
        "code": "Xianjun-Yang/DNA-GPT (demo only); paper BScore implemented here",
        "score": "BScore (n-gram overlap, n=4..25, weight n log n, length-normalised)",
        "verified": ["official demo uses gamma=0.5, temperature 0.7 (fetched); demo formula differs from paper"],
        "deviations": ["LOCAL regenerator LM instead of the source LLM", "K regenerations compute-capped"],
        "needs": "own_models",
    },
}


def get(name):
    if name not in BASELINES:
        raise KeyError(f"unknown baseline '{name}'. Known: {sorted(BASELINES)}")
    return BASELINES[name]
