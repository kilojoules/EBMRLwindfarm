"""Parse wind farm sweep logs → clean JSON + best-per-budget tables.

Logs: logs/wf_sweep{1,2,3}_17597{351,352,353}.out
Format per row:
    Budget  η    k    gs |    Power    SE   %Uncon    NegYaw    OK
"""
import re
import json
from pathlib import Path
from collections import defaultdict

LOGS = [
    "logs/wf_sweep1_17597351.out",
    "logs/wf_sweep2_17597352.out",
    "logs/wf_sweep3_17597353.out",
]
OUT = Path("results/windfarm_sweep_parsed.json")

ROW_RE = re.compile(
    r"^\s*(\d+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+\|\s+"
    r"(\d+)\s+(\d+)\s+([\d.]+)%\s+\[([-\d,\s]+)\]\s+([✓✗])"
)


def parse():
    rows = []
    for path in LOGS:
        p = Path(path)
        if not p.exists():
            print(f"missing {path}")
            continue
        for line in p.read_text().splitlines():
            m = ROW_RE.match(line)
            if not m:
                continue
            B, eta, k, gs, power, se, pct, neg, ok = m.groups()
            neg = [int(x.strip()) for x in neg.split(",")]
            rows.append({
                "budget": int(B), "eta": float(eta), "k": float(k), "gs": float(gs),
                "power_mean": int(power), "power_se": int(se),
                "pct_uncon": float(pct),
                "neg_yaw": neg,
                "max_neg": max(neg),
                "budget_satisfied": ok == "✓",
            })
    print(f"parsed {len(rows)} configs")
    return rows


def best_per_budget(rows):
    """For each budget, pick best config that satisfies (by highest power);
    if none satisfies, pick closest approach (lowest max_neg)."""
    by_B = defaultdict(list)
    for r in rows:
        by_B[r["budget"]].append(r)
    best = {}
    for B, lst in sorted(by_B.items()):
        sat = [r for r in lst if r["budget_satisfied"]]
        if sat:
            top = max(sat, key=lambda r: r["power_mean"])
            top["choice"] = "best_satisfies"
        else:
            top = min(lst, key=lambda r: r["max_neg"])
            top["choice"] = "closest_unsatisfied"
        best[B] = top
    return best


def schedule_comparison(rows):
    """Schedule 1/u^η vs exp — not in sweep data (all exp). Note."""
    return {"note": "Sweep uses exp schedule only; appendix table still uses 10-ep rerun"}


def flexibility_50eps(rows, eta=2.0, k=2.0, gs=0.1):
    """Same policy at various budgets, fixed (eta, k, gs)."""
    by_B = {}
    for r in rows:
        if r["eta"] == eta and r["k"] == k and r["gs"] == gs:
            by_B[r["budget"]] = r
    return by_B


def hard_guard_equiv(rows):
    """AC only (eta>0, gs=0.1, k=2) at budget=15 vs constant (eta=0)."""
    candidates_ac = [r for r in rows if r["budget"] == 15 and r["eta"] == 2.0
                     and r["k"] == 2.0 and r["gs"] == 0.1]
    candidates_const = [r for r in rows if r["budget"] == 15 and r["eta"] == 0.0
                        and r["k"] == 2.0 and r["gs"] == 0.1]
    return {
        "AC_eta2": candidates_ac[0] if candidates_ac else None,
        "const_eta0": candidates_const[0] if candidates_const else None,
    }


def main():
    rows = parse()
    if not rows:
        print("No data — abort")
        return
    best = best_per_budget(rows)
    flex = flexibility_50eps(rows)
    hg = hard_guard_equiv(rows)

    out = {
        "n_configs": len(rows),
        "best_per_budget": {str(B): r for B, r in best.items()},
        "flexibility_eta2_k2_gs01": {str(B): r for B, r in flex.items()},
        "hard_guard_equiv": hg,
        "schedule_note": "Sweep only has exp schedule",
        "all_rows": rows,
    }
    OUT.parent.mkdir(exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\nWrote {OUT}")

    print("\n=== Best per budget ===")
    print(f"{'B':>4s} {'eta':>5s} {'k':>4s} {'gs':>5s} "
          f"{'Power':>10s} {'%Uncon':>8s} {'NegYaw':>18s} {'Choice':>20s}")
    for B, r in best.items():
        print(f"{B:>4d} {r['eta']:>5.1f} {r['k']:>4.1f} {r['gs']:>5.2f} "
              f"{r['power_mean']:>10d} {r['pct_uncon']:>7.1f}% "
              f"{str(r['neg_yaw']):>18s} {r['choice']:>20s}")

    print("\n=== Flexibility (eta=2, k=2, gs=0.1) ===")
    for B, r in sorted(flex.items()):
        ok = "✓" if r["budget_satisfied"] else "✗"
        print(f"  B={B:>3d}: power={r['power_mean']:>10d} ({r['pct_uncon']:.1f}%) "
              f"neg={r['neg_yaw']} {ok}")

    print("\n=== Hard guard equivalent ===")
    if hg["AC_eta2"]:
        r = hg["AC_eta2"]
        print(f"  AC η=2: power={r['power_mean']}, neg={r['neg_yaw']}, "
              f"sat={r['budget_satisfied']}")
    if hg["const_eta0"]:
        r = hg["const_eta0"]
        print(f"  Const η=0: power={r['power_mean']}, neg={r['neg_yaw']}, "
              f"sat={r['budget_satisfied']}")


if __name__ == "__main__":
    main()
