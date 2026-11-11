"""
Gate a registered version and, if it passes, point production at it.

    python -m credit_risk.promote --registry gs://BUCKET/registry            # newest version
    python -m credit_risk.promote --registry gs://BUCKET/registry --version v3
    python -m credit_risk.promote --registry gs://BUCKET/registry --rollback

The running Cloud Run function notices the new champion.json within MODEL_REFRESH_SECONDS.
"""
import argparse
import os
import sys

from credit_risk import config as C
from credit_risk.registry import Registry


def gate(cand, cand_eval, champ_eval=None):
    """
    Pure decision logic: returns (approved, reasons).

    cand       candidate metrics on the fixed through-the-cycle holdout (ranking, fairness, signs)
    cand_eval  candidate metrics on the evaluation set (holdout normally, recent applications after a shock)
    champ_eval champion metrics on that same evaluation set (None if there is no champion yet)
    """
    reasons = []
    if cand["auc"] < C.MIN_AUC:
        reasons.append(f"AUC {cand['auc']} < {C.MIN_AUC}")
    if cand["ks"] < C.MIN_KS:
        reasons.append(f"KS {cand['ks']} < {C.MIN_KS}")
    if cand_eval["calibration_gap"] > C.MAX_CALIBRATION_GAP:
        reasons.append(f"calibration gap {cand_eval['calibration_gap']} > {C.MAX_CALIBRATION_GAP} on the evaluation set")
    for group in ("under_30", "62_plus"):
        air = cand.get(f"air_{group}")
        if air is not None and air < C.MIN_ADVERSE_IMPACT_RATIO:
            reasons.append(f"adverse impact ratio for {group} is {air} < {C.MIN_ADVERSE_IMPACT_RATIO}")
    for feat, expected in C.EXPECTED_SIGNS.items():
        got = cand.get("coefficient_signs", {}).get(feat)
        if got is not None and got != expected:
            reasons.append(f"risk moves the wrong way with {feat}")
    if champ_eval is not None and cand_eval["brier"] > champ_eval["brier"] - C.MIN_BRIER_IMPROVEMENT:
        reasons.append(f"Brier {cand_eval['brier']} does not beat champion {champ_eval['brier']} "
                       f"by {C.MIN_BRIER_IMPROVEMENT}")
    return not reasons, reasons


def evaluation_sets(registry, cand, current):
    """Metrics for candidate and champion on the SAME evaluation set."""
    if cand.get("eval_mode") == "recent":
        from credit_risk.data import generate
        from credit_risk.model import evaluate
        recent = generate(**cand["recent_spec"])
        champ_eval = evaluate(registry.load_bundle(current), recent) if current else None
        return cand["recent"], champ_eval, f"recent applications ({cand['recent_spec']['n']}, out-of-time)"
    return cand, (registry.metrics(current) if current else None), "fixed holdout"


def report(lines):
    text = "\n".join(lines) + "\n"
    print(text)
    if os.getenv("GITHUB_STEP_SUMMARY"):
        with open(os.environ["GITHUB_STEP_SUMMARY"], "a") as f:
            f.write(text)


def promote(registry, version=None, force=False):
    versions = registry.versions()
    if not versions:
        raise SystemExit("Registry is empty. Run `python -m credit_risk.train` first.")
    version = version or versions[-1]
    pointer = registry.champion()
    current = pointer["version"] if pointer else None
    if current == version:
        report([f"{version} is already the champion."])
        return True

    cand = registry.metrics(version)
    cand_eval, champ_eval, eval_name = evaluation_sets(registry, cand, current)
    approved, reasons = gate(cand, cand_eval, champ_eval)
    lines = [f"## Promotion decision for {version}", "",
             f"Checks on the fixed holdout: AUC {cand['auc']}, KS {cand['ks']}, "
             f"adverse impact ratio under-30 {cand['air_under_30']} / 62+ {cand['air_62_plus']}", "",
             f"Comparison on **{eval_name}**:", "", "| Metric | Candidate | Champion |", "|---|---|---|"]
    for k in ["brier", "auc", "calibration_gap", "approval_rate", "approved_default_rate"]:
        lines.append(f"| {k} | {cand_eval.get(k)} | {champ_eval.get(k) if champ_eval else '-'} |")
    lines.append("")
    if approved or force:
        registry.set_champion(version, current, "forced" if force and not approved else "passed gate")
        lines.append(f"**✅ {version} is now the champion**" + (f" (previous: {current})" if current else ""))
        if force and not approved:
            lines.append("Forced despite: " + "; ".join(reasons))
    else:
        lines.append(f"**❌ Rejected; {current or 'nothing'} stays in production.**")
        lines += [f"- {r}" for r in reasons]
    report(lines)
    return approved


def rollback(registry):
    pointer = registry.champion()
    if not pointer or not pointer.get("previous"):
        raise SystemExit("Nothing to roll back to.")
    registry.set_champion(pointer["previous"], pointer["version"], f"rollback from {pointer['version']}")
    report([f"## Rollback", "", f"**↩️ {pointer['previous']} is the champion again** (was {pointer['version']})"])


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--registry", default=None)
    p.add_argument("--version")
    p.add_argument("--rollback", action="store_true")
    p.add_argument("--force", action="store_true", help="promote even if the gate fails (recorded in history)")
    p.add_argument("--strict", action="store_true", help="exit 1 if rejected")
    a = p.parse_args()
    reg = Registry(a.registry or C.REGISTRY_URI)
    if a.rollback:
        rollback(reg)
    else:
        ok = promote(reg, a.version, a.force)
        if a.strict and not ok:
            sys.exit(1)
