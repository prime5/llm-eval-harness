"""JSON reporter — machine-readable output for CI/CD pipelines."""
import json
from datetime import datetime
from pathlib import Path


class JSONReporter:

    def write(self, results: list, path: str = "eval_results.json",
              provider: str = "") -> str:
        passed = sum(1 for r in results if r.passed)
        total = len(results)
        output = {
            "run_timestamp": datetime.utcnow().isoformat() + "Z",
            "provider": provider or "unknown",
            "summary": {
                "total": total,
                "passed": passed,
                "failed": total - passed,
                "pass_rate": round(passed / total, 4) if total else 0.0,
                "pass_rate_pct": f"{passed/total:.0%}" if total else "0%",
            },
            "results": [r.to_dict() for r in results],
        }
        Path(path).write_text(json.dumps(output, indent=2))
        return path
