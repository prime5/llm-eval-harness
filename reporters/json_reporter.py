"""JSON reporter — machine-readable output for CI/CD pipelines."""
import json
from datetime import datetime
from pathlib import Path
from evals.runner import EvalResult


class JSONReporter:

    def write(self, results: list[EvalResult], path: str = "eval_results.json") -> str:
        passed = sum(1 for r in results if r.passed)
        total = len(results)
        output = {
            "run_timestamp": datetime.utcnow().isoformat() + "Z",
            "summary": {
                "total": total,
                "passed": passed,
                "failed": total - passed,
                "pass_rate": f"{passed/total:.0%}" if total else "0%",
            },
            "results": [r.to_dict() for r in results],
        }
        Path(path).write_text(json.dumps(output, indent=2))
        return path
