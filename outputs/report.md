# NimbusAI — GPU Cost Optimization Report

**Period:** monthly  
**Baseline spend:** $27,133  
**Optimized spend:** $14,626  
**Projected savings:** $12,507  (**46%**)

## Savings by lever

| Lever | Savings (USD) |
|---|---|
| Inference (cascade/cache/batch) | $1,212 |
| Purchasing (spot/reserved) | $10,040 |
| Right-size util-lies | $655 |
| Kill idle GPUs | $600 |

## Sustainability

- Energy per query: 0.24 Wh
- Carbon per query: 0.091 gCO2e
- Cheapest+cleanest region: europe-north1

## Deep-Dive Analysis: The 'GPU-Util Lie' Mechanism

Standard telemetry like `nvidia-smi` reports **GPU-Util %** based on the percentage of time kernels are active on the GPU during sample windows. However, this measures **time-active clock, NOT compute efficiency**.

- **Underlying Hardware Mechanism:**
  1. **Memory Stalls:** Workloads with low arithmetic intensity (such as LLM decode token generation with batch size 1, where intensity is ~1-2 FLOPs/byte compared to the H100 ridge point of ~295 FLOPs/byte) spend >80% of clock cycles waiting on high-bandwidth memory (HBM3) transfers.
  2. **Kernel Launch Overhead & Warp Divergence:** Small batch sizes or fragmented tensor operations cause streaming multiprocessors (SMs) to sit idle or uncoalesced while the clock is still marked active.
- **Financial Impact:** `gpu-h100-4` shows 98.2% GPU-Util but only 19.4% MFU (Model FLOPs Utilization). NimbusAI is paying $2.50/hr ($1,800/mo) for an H100 but receiving compute throughput comparable to an underutilized card. Right-sizing this workload to an A100 saves over 28% while matching the actual compute ceiling.

## Action Plan & Prioritization by ROI

| Priority | FinOps Lever | Monthly Savings | Implementation Effort | ROI Justification |
|---|---|---|---|---|
| **P1** | **Purchasing Strategy (Spot & Commitments)** | **$10,040** (80.3% of total) | Low (Infrastructure/IaC change) | **Highest Immediate Impact.** Migrating fault-tolerant training jobs (`job-train-llm`, `job-train-embed`) to Spot with automated checkpointing and locking steady 24/7 inference to 3-year Reserved instances achieves 39.1% direct cash reduction without code refactoring. |
| **P2** | **Inference Optimization (Cascade + Cache + Batch)** | **$1,212** (9.7% of total) | Medium (App routing & API integration) | **Massive Unit Economics Gain.** Drops unit cost from $6.488 down to $1.126 per 1M tokens (82.6% reduction). Cascade routing delivers the lion's share of savings by routing 90%+ simple queries to small models. |
| **P3** | **Right-sizing Util-Lies & Memory-Bound Workloads** | **$655** (5.2% of total) | Medium (Telemetry profiling & model re-benchmarking) | Downgrading memory-bound H100 instances to A100/A10G eliminates paying for idle Tensor Cores. |
| **P4** | **Idle GPU Elimination & Automated Shutdown** | **$600** (4.8% of total) | Very Low (Kubernetes HPA / CloudWatch alarm) | Immediate stop to burning $20/day on abandoned development sandboxes and idle off-hours nodes. |

## Sustainability & Multi-Region Carbon Strategy

- **Cleanest Grid (`europe-north1`):** Emits only 30 gCO2/kWh due to abundant Nordic hydropower. Shifting the 4,227 kWh/month of interruptible training workloads to `europe-north1` cuts monthly carbon emissions by **92.1%** (-1,479.5 kg CO2e/month) compared to the baseline `us-east-1` (380 gCO2/kWh).
- **Cheapest Power (`us-east-wa`):** At $0.055/kWh and 90 gCO2/kWh, Washington state provides the lowest monthly power bill ($232.49/mo) while still reducing carbon by **76.3%**.
- **Reasoning Budget Warning:** Reasoning models consume ~80x more electrical energy per query than standard models. While accounting for only 8.4% of request volume, reasoning burns **94.0%** of total inference energy. Enforcing dynamic routing rules (triggering reasoning only when task complexity or confidence falls below 0.70) reduces energy expenditure by ~446 kWh/month.

_Figures are June-2026 as-of snapshots; re-baseline before acting._