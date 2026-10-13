"""M1 — Efficiency Audit: MFU/MBU, the GPU-Util lie, and idle waste (deck §5).

Run: python missions/m1_efficiency_audit.py
"""
from __future__ import annotations
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
from collections import defaultdict
from missions._common import load_csv, num, catalog_by_type
from finops import metrics


def run(verbose: bool = True) -> dict:
    tel = load_csv("gpu_telemetry.csv")
    cat = catalog_by_type()

    # per-row MFU/MBU, then aggregate per GPU
    agg = defaultdict(lambda: {"util": [], "mfu": [], "mbu": [], "type": None, "idle_hours": 0})
    for r in tel:
        gtype = r["gpu_type"]
        peak_fp16 = num(cat[gtype]["peak_tflops_fp16"])
        peak_bw = num(cat[gtype]["peak_bw_tbs"])
        mfu = metrics.compute_mfu(num(r["achieved_tflops"]), peak_fp16)
        mbu = metrics.compute_mbu(num(r["achieved_bw_tbs"]), peak_bw)
        a = agg[r["gpu_id"]]
        a["type"] = gtype
        a["util"].append(num(r["gpu_util_pct"]))
        a["mfu"].append(mfu)
        a["mbu"].append(mbu)
        if num(r["gpu_util_pct"]) < 10:  # effectively idle this interval (1h)
            a["idle_hours"] += 1

    summary = []
    for gid, a in agg.items():
        summary.append({
            "gpu_id": gid, "gpu_type": a["type"],
            "gpu_util_pct": round(sum(a["util"]) / len(a["util"]), 1),
            "mfu": round(sum(a["mfu"]) / len(a["mfu"]), 3),
            "mbu": round(sum(a["mbu"]) / len(a["mbu"]), 3),
            "idle_hours": a["idle_hours"],
        })

    lies = metrics.flag_util_lies(summary)
    idle_waste = 0.0
    for s in summary:
        on_demand = num(catalog_by_type()[s["gpu_type"]]["on_demand_hr"])
        idle_waste += metrics.idle_waste_usd(s["idle_hours"], on_demand)

    # --- Extension 2: Right-sizing theo MBU & $/GB-VRAM ---
    # Calculate $/GB-VRAM and $/TB/s bandwidth cost for each GPU in catalog
    cat_metrics = {}
    for gtype, c in cat.items():
        hbm = num(c["hbm_gb"])
        bw = num(c["peak_bw_tbs"])
        od = num(c["on_demand_hr"])
        cat_metrics[gtype] = {
            "cost_per_gb_vram": od / hbm if hbm > 0 else 0.0,
            "cost_per_tbs_bw": od / bw if bw > 0 else 0.0,
            "on_demand_hr": od,
            "hbm_gb": hbm,
            "peak_bw_tbs": bw,
        }

    # For memory-bound / low-MBU GPUs (MBU < 0.35), find more cost-effective right-sized replacement
    rightsize_recs = []
    total_mbu_rightsize_monthly_savings = 0.0
    for s in summary:
        cur_type = s["gpu_type"]
        cur_mbu = s["mbu"]
        # If GPU has low memory bandwidth utilization and is an expensive tier (H100/A100)
        if cur_mbu < 0.35 and cur_type in ("H100", "A100"):
            # Target GPU: A100 for H100, or A10G/L4 for A100
            target_type = "A100" if cur_type == "H100" else "A10G"
            cur_cost = cat_metrics[cur_type]["on_demand_hr"]
            target_cost = cat_metrics[target_type]["on_demand_hr"]
            hourly_savings = cur_cost - target_cost
            monthly_savings = hourly_savings * (24 - s["idle_hours"]) * 30
            pct_saved = (hourly_savings / cur_cost) * 100 if cur_cost > 0 else 0.0
            total_mbu_rightsize_monthly_savings += monthly_savings
            rightsize_recs.append({
                "gpu_id": s["gpu_id"],
                "current_type": cur_type,
                "current_mbu": cur_mbu,
                "target_type": target_type,
                "cur_vram_cost": round(cat_metrics[cur_type]["cost_per_gb_vram"], 4),
                "tgt_vram_cost": round(cat_metrics[target_type]["cost_per_gb_vram"], 4),
                "monthly_savings": round(monthly_savings, 2),
                "savings_pct": round(pct_saved, 1),
            })

    if verbose:
        print("== M1 Efficiency Audit ==")
        print(f"{'GPU':14}{'type':7}{'util%':>7}{'MFU':>7}{'MBU':>7}{'idle_h':>8}")
        for s in sorted(summary, key=lambda x: x["mfu"]):
            print(f"{s['gpu_id']:14}{s['gpu_type']:7}{s['gpu_util_pct']:>7}{s['mfu']:>7}{s['mbu']:>7}{s['idle_hours']:>8}")
        print(f"\nGPU-Util LIES (util>=90% but MFU<30%): {[l['gpu_id'] for l in lies]}")
        print(f"Idle waste (1 day): ${idle_waste:,.2f}  ->  ${idle_waste*30:,.0f}/month")

        print("\n-- [Extension 2] Right-sizing Analysis by MBU & $/GB-VRAM --")
        print(f"{'Catalog GPU':12}{'$/hr':>7}{'HBM(GB)':>9}{'Peak BW(TB/s)':>15}{'$/GB-VRAM':>11}{'$/TB/s BW':>11}")
        for gt, m in sorted(cat_metrics.items(), key=lambda x: x[1]['on_demand_hr']):
            print(f"{gt:12}${m['on_demand_hr']:>6.2f}{m['hbm_gb']:>9.0f}{m['peak_bw_tbs']:>15.2f}${m['cost_per_gb_vram']:>10.4f}${m['cost_per_tbs_bw']:>10.4f}")
        
        print("\nRight-sizing Recommendations for Low-MBU GPUs (MBU < 0.35):")
        print(f"{'GPU ID':14}{'Current':9}{'MBU':>6}{'Target':9}{'Current $/GB':>14}{'Target $/GB':>13}{'Monthly Saved':>15}{'% Saved':>9}")
        for r in rightsize_recs:
            print(f"{r['gpu_id']:14}{r['current_type']:9}{r['current_mbu']:>6.3f}{r['target_type']:9}${r['cur_vram_cost']:>13.4f}${r['tgt_vram_cost']:>12.4f}  ${r['monthly_savings']:>12,.2f}{r['savings_pct']:>8.1f}%")
        print(f"Total potential monthly MBU right-sizing savings: ${total_mbu_rightsize_monthly_savings:,.2f}/month")

    return {
        "summary": summary,
        "lies": lies,
        "idle_waste_daily": round(idle_waste, 2),
        "cat_metrics": cat_metrics,
        "rightsize_recs": rightsize_recs,
        "mbu_monthly_savings": round(total_mbu_rightsize_monthly_savings, 2),
    }


if __name__ == "__main__":
    run()
