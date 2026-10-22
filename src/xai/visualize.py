"""
HTML views of explanations (self-contained, no external assets).
Signed Integrated Gradients: red = pushes towards the explained class, blue = against it.
Attention: one-sided, grey scale.
"""

import html


def _span(tok, w, signed=True):
    a = min(1.0, abs(float(w)))
    if signed:
        color = f"rgba(220,38,38,{a:.2f})" if w > 0 else f"rgba(37,99,235,{a:.2f})"
    else:
        color = f"rgba(107,114,128,{a:.2f})"
    return f'<span title="{w:+.3f}" style="background:{color}">{html.escape(tok)}</span>'


def example_html(ex):
    """ex: dict from scripts/10_explain.py (tokens, ig_norm, attention_norm, labels, probs)."""
    toks = ex["tokens"]
    ig = "".join(_span(t, w) for t, w in zip(toks, ex["ig_norm"]))
    at = "".join(_span(t, w, signed=False) for t, w in zip(toks, ex["attention_norm"]))
    head = (f"<p><b>{html.escape(ex['row_id'])}</b> | true: {ex['label_name']} | predicted: {ex['pred_name']} "
            f"(p={ex['p_target']:.3f}) | explained class: {ex['target_name']} | IG delta: {ex['ig_delta']:.4f}</p>")
    return (f"<section style='margin-bottom:2em'>{head}<h4>Integrated Gradients</h4>"
            f"<div style='line-height:1.9;white-space:pre-wrap'>{ig}</div>"
            f"<h4>Attention ({html.escape(ex['attention_kind'])})</h4>"
            f"<div style='line-height:1.9;white-space:pre-wrap'>{at}</div></section>")


def page(title, examples):
    body = "\n".join(example_html(e) for e in examples)
    return (f"<!doctype html><html><head><meta charset='utf-8'><title>{html.escape(title)}</title></head>"
            f"<body style='font-family:system-ui,sans-serif;max-width:960px;margin:2em auto;padding:0 16px'>"
            f"<h2>{html.escape(title)}</h2><p>Red: pushes towards the explained class; blue: against it. "
            f"Hover a token for its normalised score.</p>{body}</body></html>")
