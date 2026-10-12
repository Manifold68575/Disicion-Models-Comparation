import argparse
import json

from decisionbench.metrics import monthly_tco

parser = argparse.ArgumentParser(description="Always-on deployment capacity and monthly TCO")
parser.add_argument("--monthly-requests", type=int, required=True)
parser.add_argument("--peak-rps", type=float, required=True)
parser.add_argument("--max-slo-qps", type=float, required=True)
parser.add_argument("--instance-price-per-hour", type=float, required=True)
parser.add_argument("--currency", required=True)
parser.add_argument("--hours", type=float, default=730)
parser.add_argument("--utilization", type=float, default=0.7)
parser.add_argument("--training-cost", type=float, default=0)
parser.add_argument("--maintenance-cost", type=float, default=0)
args = parser.parse_args()
result = monthly_tco(
    args.monthly_requests,
    args.peak_rps,
    args.max_slo_qps,
    args.instance_price_per_hour,
    args.hours,
    args.utilization,
    args.training_cost,
    args.maintenance_cost,
)
result["currency"] = args.currency
print(json.dumps(result, indent=2))
