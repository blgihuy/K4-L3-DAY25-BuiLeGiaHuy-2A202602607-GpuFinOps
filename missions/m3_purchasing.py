"""M3 — Purchasing Strategy: break-even, tier choice, spot-checkpoint sim (deck §4).

Run: python missions/m3_purchasing.py
"""
from __future__ import annotations
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
from missions._common import load_csv, num, catalog_by_type
from finops import pricing, sustainability

DAYS = 30


def run(verbose: bool = True) -> dict:
    jobs = load_csv("workloads.csv")
    cat = catalog_by_type()
    on_demand_monthly = optimized_monthly = 0.0
    recs = []

    # Total energy calculation for interruptible / batch jobs (Extension 5)
    interruptible_kwh_monthly = 0.0
    interruptible_jobs = []

    for j in jobs:
        gtype = j["gpu_type"]
        ngpu = int(num(j["num_gpus"]))
        hpd = num(j["hours_per_day"])
        interruptible = bool(int(num(j["interruptible"])))
        c = cat[gtype]
        gpu_hours = hpd * DAYS * ngpu
        od = num(c["on_demand_hr"])
        on_demand_cost = gpu_hours * od

        # Extension 1: recommend_tier with gpu_type and duration awareness
        tier = pricing.recommend_tier(hpd, interruptible, gpu_type=gtype, job_days=DAYS)
        if tier == "spot":
            sim = pricing.spot_checkpoint_cost(gpu_hours, num(c["spot_hr"]), od)
            opt_cost = sim["spot_cost"]
        elif tier == "reserved":
            opt_cost = gpu_hours * num(c["reserved_3yr_hr"])
        else:
            opt_cost = on_demand_cost

        on_demand_monthly += on_demand_cost
        optimized_monthly += opt_cost
        recs.append({"job_id": j["job_id"], "gpu_type": gtype, "tier": tier,
                     "on_demand": round(on_demand_cost), "optimized": round(opt_cost)})

        # Track energy for interruptible batch/training jobs
        if interruptible:
            watts = num(c.get("watts", 400))
            kwh = (gpu_hours * watts) / 1000.0
            interruptible_kwh_monthly += kwh
            interruptible_jobs.append({"job_id": j["job_id"], "gpu": gtype, "gpu_hours": gpu_hours, "kwh": kwh})

    savings = on_demand_monthly - optimized_monthly
    savings_pct = savings / on_demand_monthly * 100 if on_demand_monthly else 0.0

    # Extension 5: Multi-region Carbon & Electricity Cost comparison
    region_comparison = {}
    baseline_region = "us-east-1"
    base_kwh_price = sustainability.REGION_PRICE_KWH[baseline_region]
    base_carbon_factor = sustainability.REGION_CARBON[baseline_region]
    base_elec_cost = interruptible_kwh_monthly * base_kwh_price
    base_carbon_kg = (interruptible_kwh_monthly * base_carbon_factor) / 1000.0

    for reg, carbon_factor in sustainability.REGION_CARBON.items():
        price_kwh = sustainability.REGION_PRICE_KWH.get(reg, 0.10)
        elec_cost = interruptible_kwh_monthly * price_kwh
        carbon_kg = (interruptible_kwh_monthly * carbon_factor) / 1000.0
        saved_carbon_kg = base_carbon_kg - carbon_kg
        pct_carbon_saved = (saved_carbon_kg / base_carbon_kg) * 100 if base_carbon_kg > 0 else 0.0
        region_comparison[reg] = {
            "price_kwh": price_kwh,
            "carbon_g_per_kwh": carbon_factor,
            "monthly_elec_usd": round(elec_cost, 2),
            "monthly_carbon_kg": round(carbon_kg, 1),
            "carbon_saved_kg": round(saved_carbon_kg, 1),
            "carbon_saved_pct": round(pct_carbon_saved, 1),
        }

    cleanest_reg = min(region_comparison.items(), key=lambda x: x[1]["monthly_carbon_kg"])[0]
    cheapest_reg = min(region_comparison.items(), key=lambda x: x[1]["monthly_elec_usd"])[0]

    if verbose:
        print("== M3 Purchasing Strategy ==")
        print(f"break-even utilization @ 45% reserved discount = {pricing.break_even_utilization(0.45):.0%}")
        print(f"{'job':18}{'gpu':7}{'tier':11}{'on-demand':>12}{'optimized':>12}")
        for r in recs:
            print(f"{r['job_id']:18}{r['gpu_type']:7}{r['tier']:11}${r['on_demand']:>11,}${r['optimized']:>11,}")
        print(f"\nmonthly: on-demand ${on_demand_monthly:,.0f} -> optimized ${optimized_monthly:,.0f}  ({savings_pct:.1f}% saved)")

        print("\n-- [Extension 5] Carbon-aware Scheduling Analysis (Interruptible Jobs) --")
        print(f"Total interruptible compute: {interruptible_kwh_monthly:,.1f} kWh/month across {len(interruptible_jobs)} batch/training jobs.")
        print(f"{'Region':18}{'$/kWh':>8}{'gCO2/kWh':>10}{'Elec Cost/mo':>14}{'CO2 (kg/mo)':>13}{'CO2 Saved %':>13}")
        for reg, data in sorted(region_comparison.items(), key=lambda x: x[1]["monthly_carbon_kg"]):
            flag = " [CLEANEST]" if reg == cleanest_reg else (" [CHEAPEST]" if reg == cheapest_reg else "")
            print(f"{reg:18}${data['price_kwh']:>7.3f}{data['carbon_g_per_kwh']:>10}${data['monthly_elec_usd']:>13,.2f}{data['monthly_carbon_kg']:>13,.1f}{data['carbon_saved_pct']:>12.1f}%{flag}")

        print(f"\nRecommendation: Migrate interruptible training jobs ({', '.join(j['job_id'] for j in interruptible_jobs[:3])}...)")
        print(f"  -> To '{cleanest_reg}': cuts carbon footprint by {region_comparison[cleanest_reg]['carbon_saved_pct']:.1f}% (-{region_comparison[cleanest_reg]['carbon_saved_kg']:,.1f} kg CO2e/month) with electricity cost ${region_comparison[cleanest_reg]['monthly_elec_usd']:,.2f}/mo.")
        print(f"  -> To '{cheapest_reg}': minimizes power bill to ${region_comparison[cheapest_reg]['monthly_elec_usd']:,.2f}/mo (-{(1 - region_comparison[cheapest_reg]['monthly_elec_usd']/base_elec_cost)*100:.1f}%) and cuts carbon by {region_comparison[cheapest_reg]['carbon_saved_pct']:.1f}%.")
        print(f"  -> Latency Trade-off: Batch/training jobs are latency-insensitive; cross-region data transfer is a one-time ingress cost offset by massive power & carbon savings.")

    return {
        "recommendations": recs,
        "on_demand_monthly": round(on_demand_monthly),
        "optimized_monthly": round(optimized_monthly),
        "savings_pct": round(savings_pct, 1),
        "interruptible_kwh_monthly": round(interruptible_kwh_monthly, 1),
        "region_comparison": region_comparison,
        "cleanest_region": cleanest_reg,
        "cheapest_region": cheapest_reg,
    }


if __name__ == "__main__":
    run()

