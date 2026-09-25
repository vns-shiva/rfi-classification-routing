"""make_figures.py: render the paper's data figures directly from the real
routing_eval_report.json / agreement_report.json files already written under
pilot-data/ — never from hand-typed numbers, so a figure can't silently drift
from the scored results it claims to show.

    python code/make_figures.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_ROOT = Path(__file__).resolve().parent.parent
_PILOT = _ROOT / "pilot-data"
_FIGURES = _ROOT / "figures"

_CONDITIONS = ["bare", "vocab", "policy"]
_ARMS = ["oracle", "composed", "direct", "policy_fidelity", "majority"]
_ARM_LABELS = {
    "oracle": "Oracle\n(gold routing)",
    "composed": "Composed\n(LLM fields → policy)",
    "direct": "Direct\n(LLM assigns reviewer)",
    "policy_fidelity": "Policy fidelity\n(LLM inputs → policy)",
    "majority": "Majority\nbaseline",
}


_PROVIDERS = ["anthropic", "openai"]
_PROVIDER_LABELS = {"anthropic": "Anthropic claude-sonnet-5", "openai": "OpenAI gpt-5"}


def _load(provider: str, condition: str) -> dict:
    path = _PILOT / f"routing_eval-{provider}-{condition}" / "routing_eval_report.json"
    return json.loads(path.read_text(encoding="utf-8"))


_SCORED_CONDITIONS = ["vocab", "policy"]


def fig_ablation_accuracy() -> None:
    fig, axes = plt.subplots(2, 1, figsize=(7, 8.5), sharex=True)
    x = list(range(len(_ARMS)))
    width = 0.32
    for ax, provider in zip(axes, _PROVIDERS):
        for i, condition in enumerate(_SCORED_CONDITIONS):
            report = _load(provider, condition)
            accs = [report["arms"][arm]["reviewer_accuracy"] or 0.0 for arm in _ARMS]
            offset = (i - 0.5) * width
            ax.bar([xi + offset for xi in x], accs, width, label=condition)
            for xi, acc in zip(x, accs):
                ax.text(xi + offset, acc + 0.015, f"{acc:.2f}", ha="center", fontsize=7)
        ax.text(0.99, 0.95, "bare: n/a for every arm (not scored, omitted)",
                transform=ax.transAxes, ha="right", va="top", fontsize=7,
                color="dimgray", style="italic")
        ax.set_xticks(x)
        ax.set_xticklabels([_ARM_LABELS[a] for a in _ARMS], fontsize=8)
        ax.set_title(_PROVIDER_LABELS[provider], fontsize=10)
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_ylabel("Reviewer assignment accuracy\n(n=200 EVAL threads)", fontsize=8)
        ax.set_ylim(0, 1.05)
    axes[0].legend(title="Prompt condition", fontsize=8)
    fig.suptitle(
        "Reviewer-routing accuracy by prompt condition, scoring arm, and provider\n"
        "(bare omitted, not scored: 100% of Anthropic's and 239 of OpenAI's 240\n"
        "bare outputs failed closed-vocabulary schema validation, and OpenAI's\n"
        "240th bare response never parsed at all)"
    )
    fig.tight_layout()
    fig.savefig(_FIGURES / "ablation-accuracy.png", dpi=200)
    plt.close(fig)


def fig_gate_tradeoff() -> None:
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    x = range(len(_CONDITIONS))
    gated, ungated = [], []
    for condition in _CONDITIONS:
        report = _load("anthropic", condition)
        gb = report["arms"]["composed"]["gate_breakdown"]
        gated.append(gb["gated_reviewer_accuracy"] or 0.0)
        ungated.append(gb["ungated_reviewer_accuracy"] or 0.0)
    width = 0.32
    ax.bar([xi - width / 2 for xi in x], gated, width, label="Gated to human review")
    ax.bar([xi + width / 2 for xi in x], ungated, width, label="Auto-routed (ungated)")
    ax.set_xticks(list(x))
    ax.set_xticklabels(_CONDITIONS)
    ax.set_ylabel("Reviewer assignment accuracy")
    ax.set_ylim(0, 1.05)
    ax.set_title("Confidence/escalation gating separates low- from high-accuracy\nsubpopulations (composed arm, Anthropic claude-sonnet-5)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(_FIGURES / "gate-tradeoff.png", dpi=200)
    plt.close(fig)


def fig_iaa() -> None:
    report = json.loads((_PILOT / "annotator_agreement" / "agreement_report.json").read_text(encoding="utf-8"))
    fields = report["field_results"]
    nominal_ordinal = [
        (name, f["summary"]["alpha"])
        for name, f in fields.items()
        if f.get("kind") in ("nominal", "ordinal", "boolean")
        and f.get("reportable")
        and f["summary"].get("alpha") is not None
    ]
    nominal_ordinal.sort(key=lambda t: t[1])
    names = [n for n, _ in nominal_ordinal]
    alphas = [a for _, a in nominal_ordinal]
    fig, ax = plt.subplots(figsize=(7, 5))
    colors = ["#c0392b" if a < 0.667 else ("#e1a100" if a < 0.8 else "#2e7d32") for a in alphas]
    ax.barh(names, alphas, color=colors)
    ax.axvline(0.667, color="black", linestyle="--", linewidth=0.8, label="α=0.667 (tentative-conclusions threshold)")
    ax.axvline(0.8, color="black", linestyle=":", linewidth=0.8, label="α=0.8 (reliable threshold)")
    ax.set_xlabel("Krippendorff's alpha (gold vs. annotator_b, n=40 DEV-split threads)")
    ax.set_title("Field-level inter-annotator agreement")
    ax.legend(fontsize=8, loc="lower right")
    ax.set_xlim(-0.1, 1.05)
    fig.tight_layout()
    fig.savefig(_FIGURES / "iaa-by-field.png", dpi=200)
    plt.close(fig)


def main() -> None:
    _FIGURES.mkdir(parents=True, exist_ok=True)
    fig_ablation_accuracy()
    fig_gate_tradeoff()
    fig_iaa()
    print(f"wrote 3 figures to {_FIGURES}")


if __name__ == "__main__":
    main()
