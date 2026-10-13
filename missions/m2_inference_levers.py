"""M2 — Inference Cost Levers: $/1M-token, batch x cache x cascade (deck §7).

Run: python missions/m2_inference_levers.py
"""
from __future__ import annotations
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
from missions._common import load_csv, num
from finops import pricing

from finops import pricing, sustainability

# $/1M tokens (input, output) — illustrative 2026.
MODEL_PRICES = {"small": (0.20, 0.40), "large": (3.00, 15.00)}


def run(verbose: bool = True) -> dict:
    rows = load_csv("token_usage.csv")
    base_cost = opt_cost = 0.0
    total_tokens = 0

    # Extension 3 & 4 tracking structures
    reasoning_counts = {0: 0, 1: 0}
    reasoning_tokens = {0: 0, 1: 0}
    reasoning_cost = {0: 0.0, 1: 0.0}
    reasoning_wh = {0: 0.0, 1: 0.0}

    # Evaluate cache feasibility (Extension 3)
    # Default assumptions: prefix write cost = 1.0x input price, read discount = 90% (0.1x)
    # Average reuse across synthetic prompt logs estimated at ~2.5 reads
    avg_cache_reads = 2.5
    cache_effective = {}
    for tier, (pin, _) in MODEL_PRICES.items():
        be_reads = pricing.break_even_cache_reads(pin, pin, read_discount=0.10)
        is_worth = pricing.cache_is_worth_it(avg_cache_reads, pin, read_discount=0.10, base_price_per_m=pin)
        cache_effective[tier] = {"break_even_reads": round(be_reads, 2), "is_worth": is_worth}

    for r in rows:
        inp, out = int(num(r["input_tokens"])), int(num(r["output_tokens"]))
        cached = int(num(r["cached_input_tokens"]))
        is_batch = bool(int(num(r["is_batch"])))
        is_reason = int(num(r.get("is_reasoning", 0)))
        req_tokens = inp + out
        total_tokens += req_tokens

        # BASELINE: naive deployment — everything on the large model, no cache, no batch
        lin, lout = MODEL_PRICES["large"]
        req_base = pricing.request_cost(inp, out, lin, lout)
        base_cost += req_base

        # OPTIMIZED: cascade (route_tier), prompt caching (if worth it), batch API
        tier = r["route_tier"]
        pin, pout = MODEL_PRICES[tier]
        use_cache = cached if cache_effective.get(tier, {}).get("is_worth", True) else 0
        req_opt = pricing.request_cost(inp, out, pin, pout, cached_in=use_cache, batch=is_batch)
        opt_cost += req_opt

        # Track reasoning metrics (Extension 4)
        req_wh = sustainability.wh_per_query(req_tokens, is_reasoning=bool(is_reason))
        reasoning_counts[is_reason] += 1
        reasoning_tokens[is_reason] += req_tokens
        reasoning_cost[is_reason] += req_opt
        reasoning_wh[is_reason] += req_wh

    base_pm = pricing.dollars_per_million(base_cost, total_tokens)
    opt_pm = pricing.dollars_per_million(opt_cost, total_tokens)
    savings_pct = (1 - opt_cost / base_cost) * 100 if base_cost else 0.0

    total_wh = sum(reasoning_wh.values())
    total_opt_cost = sum(reasoning_cost.values())

    if verbose:
        print("== M2 Inference Cost Levers ==")
        print(f"requests={len(rows)}  tokens={total_tokens:,}")
        print(f"baseline  : ${base_cost:,.2f}/day   ${base_pm:.3f}/1M-token")
        print(f"optimized : ${opt_cost:,.2f}/day   ${opt_pm:.3f}/1M-token")
        print(f"savings   : {savings_pct:.1f}%  (cascade + caching + batch)")
        print(f"discount stack (batch + 100% cache): {pricing.discount_stack(batch=True, cache_hit_frac=1.0):.3f} of naive")

        print("\n-- [Extension 3] Prompt Caching Break-even Analysis --")
        for tier, ce in cache_effective.items():
            print(f"  Tier {tier:6}: break-even = {ce['break_even_reads']} reads/prefix | Worth it @ {avg_cache_reads} avg reads? {ce['is_worth']}")

        print("\n-- [Extension 4] Reasoning Workload & Energy Analysis --")
        print(f"{'Category':12}{'Requests':>10}{'% Req':>8}{'Tokens':>12}{'% Tok':>8}{'Cost/Day':>10}{'% Cost':>8}{'Energy(Wh)':>12}{'% Energy':>9}")
        for k, name in [(0, "Standard"), (1, "Reasoning")]:
            req_pct = (reasoning_counts[k] / len(rows)) * 100
            tok_pct = (reasoning_tokens[k] / total_tokens) * 100
            cost_pct = (reasoning_cost[k] / total_opt_cost) * 100 if total_opt_cost else 0
            wh_pct = (reasoning_wh[k] / total_wh) * 100 if total_wh else 0
            print(f"{name:12}{reasoning_counts[k]:>10}{req_pct:>7.1f}%{reasoning_tokens[k]:>12,}{tok_pct:>7.1f}%${reasoning_cost[k]:>9.2f}{cost_pct:>7.1f}%{reasoning_wh[k]:>12.1f}{wh_pct:>8.1f}%")

        # Routing recommendation
        print(f"\nInsight: Reasoning is only {reasoning_counts[1]/len(rows)*100:.1f}% of requests, but burns {reasoning_wh[1]/total_wh*100:.1f}% of inference energy (~80x multiplier)!")
        capped_savings_wh = reasoning_wh[1] * 0.5
        capped_savings_usd = reasoning_cost[1] * 0.5 * 30
        print(f"Recommendation: Restrict reasoning to confidence < 0.70. A 50% cut in reasoning saves {capped_savings_wh:.1f} Wh/day and ${capped_savings_usd:.2f}/month.")

    return {
        "baseline_daily": round(base_cost, 2), "optimized_daily": round(opt_cost, 2),
        "baseline_per_m": round(base_pm, 3), "optimized_per_m": round(opt_pm, 3),
        "savings_pct": round(savings_pct, 1), "total_tokens": total_tokens,
        "cache_analysis": cache_effective,
        "reasoning_counts": reasoning_counts,
        "reasoning_wh": {k: round(v, 2) for k, v in reasoning_wh.items()},
        "reasoning_cost": {k: round(v, 2) for k, v in reasoning_cost.items()},
    }


if __name__ == "__main__":
    run()

