"""
╔══════════════════════════════════════════════════════════════════════╗
║  TRIADS — Interactive Alloy Yield Strength Predictor                ║
║  Gradio App for the TRIADS V13A SOTA Ensemble                       ║
║                                                                      ║
║  Run locally:   python app.py                                        ║
║  HF Spaces:     Auto-detected and hosted                             ║
╚══════════════════════════════════════════════════════════════════════╝
"""

import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import torch
import gradio as gr
from pymatgen.core import Composition

from model_arch import DeepHybridTRM, ExpandedFeaturizer


# ══════════════════════════════════════════════════════════════════════
# 1. GLOBAL MODEL LOADING
# ══════════════════════════════════════════════════════════════════════

print("⚙️  Initializing TRIADS V13A Ensemble...")

CKPT_PATH = "triads_v13a_ensemble.pt"

# Try loading locally first, then from HuggingFace
if not os.path.exists(CKPT_PATH):
    try:
        from huggingface_hub import hf_hub_download
        print("   Downloading checkpoint from HuggingFace...")
        CKPT_PATH = hf_hub_download(
            repo_id="Rtx09/TRIADS",
            filename="triads_v13a_ensemble.pt"
        )
    except Exception as e:
        raise FileNotFoundError(
            f"Could not find or download checkpoint: {e}")

ckpt = torch.load(CKPT_PATH, map_location="cpu")
CONFIG = ckpt["config"]
SEEDS = ckpt["seeds"]
N_MODELS = ckpt["n_models"]

# Load all 25 models
MODELS = []
for key, state_dict in ckpt["ensemble_weights"].items():
    # Force n_extra=200 to match pool.0.weight shape [96, 264] (64+200=264)
    m = DeepHybridTRM(n_extra=200)
    m.load_state_dict(state_dict)
    m.eval()
    MODELS.append(m)

print(f"   ✓ Loaded {len(MODELS)} models ({N_MODELS} expected)")
print(f"   ✓ Architecture: {ckpt['model_name']} ({sum(p.numel() for p in MODELS[0].parameters()):,} params)")

# Initialize featurizer (downloads Mat2Vec on first run)
print("   Loading featurizer (Magpie + Mat2Vec + Matminer)...")
FEATURIZER = ExpandedFeaturizer()
print("   ✓ Featurizer ready\n")


# ══════════════════════════════════════════════════════════════════════
# 2. PREDICTION LOGIC
# ══════════════════════════════════════════════════════════════════════

def predict_yield_strength(formula: str):
    """
    Full ensemble prediction pipeline.
    Returns: prediction text, per-model stats, composition breakdown.
    """
    if not formula or not formula.strip():
        return (
            "⚠️ Please enter a chemical composition.",
            "",
            ""
        )

    formula = formula.strip()

    # ── Parse composition ─────────────────────────────────────────
    try:
        comp = Composition(formula)
    except Exception as e:
        return (
            f"❌ Invalid composition: `{formula}`\n\n"
            f"Error: {str(e)}\n\n"
            f"**Tips:**\n"
            f"- Use element symbols: `Fe`, `Cr`, `Ni`, `C`, etc.\n"
            f"- Fractions must sum to ~1: `Fe0.7Cr0.2Ni0.1`\n"
            f"- Or use integer counts: `Fe70Cr20Ni10`",
            "",
            ""
        )

    # ── Composition breakdown ─────────────────────────────────────
    elements = comp.get_el_amt_dict()
    total = sum(elements.values())
    comp_lines = []
    for el, amt in sorted(elements.items(), key=lambda x: -x[1]):
        pct = (amt / total) * 100
        bar = "█" * int(pct / 2) + "░" * (50 - int(pct / 2))
        comp_lines.append(f"**{el:>3s}** `{bar}` {pct:5.1f}%")
    comp_breakdown = "### 🧪 Composition Breakdown\n\n" + "\n\n".join(comp_lines)

    # ── Featurize ─────────────────────────────────────────────────
    try:
        X = FEATURIZER.featurize_all([comp])
        X_tensor = torch.tensor(X, dtype=torch.float32)
    except Exception as e:
        return (
            f"❌ Featurization failed for `{formula}`:\n{str(e)}",
            "",
            comp_breakdown
        )

    # ── Ensemble prediction ───────────────────────────────────────
    all_preds = []
    with torch.no_grad():
        for model in MODELS:
            pred = model(X_tensor).item()
            all_preds.append(pred)

    all_preds = np.array(all_preds)
    ensemble_mean = np.mean(all_preds)
    ensemble_std = np.std(all_preds)
    pred_min = np.min(all_preds)
    pred_max = np.max(all_preds)

    # ── Format results ────────────────────────────────────────────
    result = (
        f"# 🎯 {ensemble_mean:.1f} MPa\n\n"
        f"**Predicted Yield Strength** for `{comp.reduced_formula}`\n\n"
        f"---\n\n"
        f"### 📊 Ensemble Statistics\n\n"
        f"| Metric | Value |\n"
        f"|:-------|------:|\n"
        f"| **Ensemble Mean** | **{ensemble_mean:.2f} MPa** |\n"
        f"| Ensemble Std Dev | ±{ensemble_std:.2f} MPa |\n"
        f"| Range | {pred_min:.2f} – {pred_max:.2f} MPa |\n"
        f"| Models Used | {len(all_preds)} |\n\n"
        f"---\n\n"
        f"### 🔍 Confidence\n\n"
    )

    # Confidence assessment based on ensemble agreement
    cv = (ensemble_std / abs(ensemble_mean)) * 100 if ensemble_mean != 0 else 100
    if cv < 3:
        result += f"🟢 **High confidence** — models strongly agree (CV = {cv:.1f}%)"
    elif cv < 8:
        result += f"🟡 **Moderate confidence** — some model disagreement (CV = {cv:.1f}%)"
    else:
        result += f"🔴 **Low confidence** — significant model disagreement (CV = {cv:.1f}%)\n\n> This composition may be outside the training distribution."

    # ── Per-seed breakdown ────────────────────────────────────────
    seed_lines = ["### 🌱 Per-Seed Predictions\n"]
    seed_lines.append("| Seed | Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 | **Avg** |")
    seed_lines.append("|:-----|-------:|-------:|-------:|-------:|-------:|--------:|")
    for si, seed in enumerate(SEEDS):
        fold_preds = all_preds[si * 5 : (si + 1) * 5]
        avg = np.mean(fold_preds)
        vals = " | ".join(f"{p:.1f}" for p in fold_preds)
        seed_lines.append(f"| {seed} | {vals} | **{avg:.1f}** |")
    seed_breakdown = "\n".join(seed_lines)

    return result, seed_breakdown, comp_breakdown


# ══════════════════════════════════════════════════════════════════════
# 3. GRADIO INTERFACE
# ══════════════════════════════════════════════════════════════════════

EXAMPLES = [
    ["Fe0.7Cr0.15Ni0.15"],
    ["Fe0.8C0.005Mn0.01Cr0.12Ni0.065"],
    ["Fe0.9Cr0.05Mo0.03V0.02"],
    ["Fe0.85Cr0.1Ni0.05"],
    ["Fe0.6Cr0.2Ni0.1Mo0.05Mn0.05"],
    ["Fe0.95C0.01Si0.02Mn0.02"],
]

DESCRIPTION = """
<div style="text-align: center; max-width: 800px; margin: auto;">
<p style="font-size: 1.1em;">
A <strong>224K-parameter</strong> deep learning model achieving <strong>91.20 MPa MAE</strong> on the 
<a href="https://matbench.materialsproject.org/" target="_blank">Matbench Steels</a> benchmark — 
surpassing CrabNet, Darwin, and Random Forest baselines.
</p>
<p style="font-size: 0.95em; color: #888;">
Architecture: 2-Layer Self-Attention → Recursive MLP (20 steps) → Deep Supervision | 5-Seed Ensemble (25 models)
<br>
<a href="https://github.com/Rtx09x/TRIADS" target="_blank">📄 Paper & Code on GitHub</a> · 
<a href="https://huggingface.co/Rtx09/TRIADS" target="_blank">🤗 Model on HuggingFace</a>
</p>
</div>
"""

ARTICLE = """
<div style="text-align: center; margin-top: 20px; padding: 20px; background: rgba(128,128,128,0.05); border-radius: 12px;">
<h3>How it works</h3>
<p>
<strong>1. Featurization:</strong> Your composition is converted into ~462 chemical features 
(Magpie descriptors + Mat2Vec embeddings + Matminer descriptors).<br>
<strong>2. Attention:</strong> Two self-attention layers learn property interactions across 22 chemical property tokens.<br>
<strong>3. Recursive Reasoning:</strong> A shared-weight MLP refines the prediction over 20 iterative steps.<br>
<strong>4. Ensemble:</strong> 25 independently trained models (5 seeds × 5 folds) are averaged for the final prediction.
</p>
<p style="font-size: 0.85em; color: #888;">
Trained on the matbench_steels dataset (312 steel compositions). 
Predictions are most reliable for compositions within the training distribution.
<br><br>
Built by <a href="https://github.com/Rtx09x" target="_blank">Rudra Tiwari</a> · 
Full research journey and ablation studies on <a href="https://github.com/Rtx09x/TRIADS" target="_blank">GitHub</a>
</p>
</div>
"""

CSS = """
.gradio-container {
    max-width: 1100px !important;
    margin: auto !important;
}
h1 {
    text-align: center;
    font-size: 2.2em !important;
    margin-bottom: 0 !important;
}
"""

with gr.Blocks(
    title="TRIADS — Alloy Yield Strength Predictor",
    theme=gr.themes.Soft(
        primary_hue="emerald",
        secondary_hue="blue",
        neutral_hue="slate",
        font=gr.themes.GoogleFont("Inter"),
    ),
    css=CSS,
) as demo:

    gr.Markdown("# ⚛️ TRIADS Yield Strength Predictor")
    gr.HTML(DESCRIPTION)

    with gr.Row():
        with gr.Column(scale=1):
            formula_input = gr.Textbox(
                label="Chemical Composition",
                placeholder="e.g., Fe0.7Cr0.15Ni0.15",
                info="Enter a steel alloy formula using element symbols and fractions.",
                lines=1,
                max_lines=1,
            )
            predict_btn = gr.Button(
                "🔬 Predict Yield Strength",
                variant="primary",
                size="lg",
            )
            gr.Examples(
                examples=EXAMPLES,
                inputs=formula_input,
                label="Example Compositions",
            )

        with gr.Column(scale=2):
            result_output = gr.Markdown(
                label="Prediction",
                value="*Enter a composition and click predict...*",
            )

    with gr.Row():
        with gr.Column():
            comp_output = gr.Markdown(label="Composition")
        with gr.Column():
            seed_output = gr.Markdown(label="Per-Seed Details")

    gr.HTML(ARTICLE)

    # Wire up
    predict_btn.click(
        fn=predict_yield_strength,
        inputs=[formula_input],
        outputs=[result_output, seed_output, comp_output],
    )
    formula_input.submit(
        fn=predict_yield_strength,
        inputs=[formula_input],
        outputs=[result_output, seed_output, comp_output],
    )


if __name__ == "__main__":
    demo.launch(share=False)
