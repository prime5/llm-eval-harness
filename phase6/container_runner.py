"""
Phase 6 — Containerized Test Runner

Runs each eval suite in an isolated Docker container, executes suites in
parallel, and aggregates JSON results into a combined report.

Why containers?
  - Environment parity: identical Python + deps on every run, no "works on
    my machine" drift between CI and local.
  - Isolation: a crashing or hung suite can't affect the others.
  - Parallelism: suites run concurrently (one container each), cutting
    wall-clock time proportionally to the number of suites.
  - K8s-ready: the same isolation model maps directly to Kubernetes Jobs —
    each container becomes a pod, the orchestrator becomes a Job controller.

Usage:
    # Run default suites in containers
    python -m phase6.container_runner

    # Run specific suites
    python -m phase6.container_runner --suites test_cases/basic_qa.yaml test_cases/hallucination.yaml

    # Run all suites
    python -m phase6.container_runner --all

    # Use a pre-built image (skip local build)
    python -m phase6.container_runner --image myrepo/eval-harness:latest

    # Set parallelism (default: all suites at once)
    python -m phase6.container_runner --max-parallel 3

    # Fail if any suite pass rate drops below threshold
    python -m phase6.container_runner --min-pass-rate 0.8
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
load_dotenv()  # load .env before reading OPENAI_API_KEY from os.environ

try:
    import docker
    from docker.errors import BuildError, ContainerError, ImageNotFound
    DOCKER_AVAILABLE = True
except ImportError:
    DOCKER_AVAILABLE = False


# ── Constants ─────────────────────────────────────────────────────────────────

IMAGE_NAME = "eval-harness"
IMAGE_TAG = "phase6"
FULL_IMAGE = f"{IMAGE_NAME}:{IMAGE_TAG}"

DEFAULT_SUITES = [
    "test_cases/basic_qa.yaml",
    "test_cases/instruction_following.yaml",
    "test_cases/complex_explanations.yaml",
]

ALL_SUITES = DEFAULT_SUITES + [
    "test_cases/prompt_injection.yaml",
    "test_cases/hallucination.yaml",
    "test_cases/safety_compliance.yaml",
    "test_cases/jailbreak.yaml",
    "test_cases/tool_calling.yaml",
    "test_cases/multi_turn.yaml",
]


# ── Data classes ──────────────────────────────────────────────────────────────

@dataclass
class ContainerResult:
    suite: str
    exit_code: int
    stdout: str
    stderr: str
    duration_s: float
    results: list[dict] = field(default_factory=list)
    error: Optional[str] = None

    @property
    def passed(self) -> bool:
        return self.exit_code == 0 and self.error is None

    @property
    def suite_pass_rate(self) -> float:
        if not self.results:
            return 0.0
        return sum(1 for r in self.results if r.get("passed")) / len(self.results)


# ── Core runner ───────────────────────────────────────────────────────────────

class ContainerRunner:
    """
    Orchestrates running eval suites in Docker containers.

    Each call to run_suite() spins up a fresh container, injects the suite
    path and API key via environment variables, waits for completion, and
    parses the JSON results from stdout.
    """

    def __init__(
        self,
        image: str = FULL_IMAGE,
        harness_root: Optional[Path] = None,
        api_key: Optional[str] = None,
        provider: str = "openai",
        timeout_s: int = 300,
    ):
        if not DOCKER_AVAILABLE:
            raise RuntimeError(
                "docker Python SDK not installed. Run: pip install docker"
            )

        self._client = docker.from_env()
        self._image = image
        self._harness_root = harness_root or Path(__file__).parent.parent
        self._api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        self._provider = provider
        self._timeout_s = timeout_s

    # ── Public API ────────────────────────────────────────────────────────────

    def build_image(self, nocache: bool = False) -> str:
        """Build the harness Docker image from the project Dockerfile."""
        print(f"🔨 Building image {self._image} ...")
        t0 = time.time()
        try:
            image, logs = self._client.images.build(
                path=str(self._harness_root),
                tag=self._image,
                nocache=nocache,
                rm=True,
            )
            elapsed = time.time() - t0
            print(f"   ✓ Built in {elapsed:.1f}s  (id={image.short_id})")
            return image.id
        except BuildError as exc:
            raise RuntimeError(f"Docker build failed: {exc}") from exc

    def ensure_image(self) -> None:
        """Build image if it doesn't exist locally.

        Uses `docker images -q` via the low-level API rather than images.get(),
        which can fail on multi-platform manifest lists built by recent Docker.
        """
        images = self._client.images.list(name=self._image)
        if images:
            print(f"   ✓ Image {self._image} found locally")
        else:
            self.build_image()

    def run_suite(self, suite_path: str) -> ContainerResult:
        """
        Run a single eval suite in an isolated container.

        Results strategy: mount a per-suite host directory into the container
        at /results. run_evals.py writes eval_results.json there. After the
        container exits we read the file directly from the host — no stdout
        parsing required, no data lost to terminal formatting.

        Volume path is under phase6/.results/ (within the harness root, so
        it's always inside Docker Desktop's /Users file-sharing scope on macOS).
        """
        t0 = time.time()

        # Per-suite output directory on the host (Docker Desktop shares /Users)
        suite_slug = Path(suite_path).stem
        host_results_dir = self._harness_root / "phase6" / ".results" / suite_slug
        host_results_dir.mkdir(parents=True, exist_ok=True)

        env = {
            "OPENAI_API_KEY": self._api_key,
            "ANTHROPIC_API_KEY": os.environ.get("ANTHROPIC_API_KEY", ""),
        }

        cmd = [
            "--suite", suite_path,
            "--provider", self._provider,
            "--format", "json",
            "--output-dir", "/results",   # write JSON into the mounted volume
            "--min-pass-rate", "0.0",
            "--variance-runs", "1",
        ]

        volumes = {
            str(host_results_dir): {"bind": "/results", "mode": "rw"}
        }

        try:
            container = self._client.containers.run(
                image=self._image,
                command=cmd,
                environment=env,
                volumes=volumes,
                detach=True,
                remove=False,
                mem_limit="512m",
                network_mode="bridge",
            )

            try:
                result = container.wait(timeout=self._timeout_s)
                exit_code = result.get("StatusCode", 1)
                logs = container.logs(stdout=True, stderr=True).decode("utf-8", errors="replace")
            finally:
                container.remove(force=True)

            duration_s = time.time() - t0

            # Read results from the mounted volume file
            results_file = host_results_dir / "eval_results.json"
            results, parse_error = self._read_results_file(results_file)

            error = None
            if exit_code != 0:
                # Surface last few lines of logs as the error message
                error = "\n".join(logs.strip().splitlines()[-5:])
            elif parse_error:
                error = parse_error

            return ContainerResult(
                suite=suite_path,
                exit_code=exit_code,
                stdout=logs,
                stderr="",
                duration_s=duration_s,
                results=results,
                error=error,
            )

        except Exception as exc:
            return ContainerResult(
                suite=suite_path,
                exit_code=1,
                stdout="",
                stderr=str(exc),
                duration_s=time.time() - t0,
                error=str(exc),
            )

    def run_parallel(
        self,
        suites: list[str],
        max_parallel: Optional[int] = None,
    ) -> list[ContainerResult]:
        """
        Run multiple suites in parallel containers.

        max_parallel caps concurrency (default: all suites at once).
        Results are returned in the same order as input suites.
        """
        workers = max_parallel or len(suites)
        ordered: dict[str, ContainerResult] = {}

        print(f"\n🚀 Launching {len(suites)} container(s) (max_parallel={workers})")
        print("   " + ", ".join(Path(s).stem for s in suites))
        print()

        with ThreadPoolExecutor(max_workers=workers) as pool:
            future_to_suite = {
                pool.submit(self._run_and_report, s): s
                for s in suites
            }
            for future in as_completed(future_to_suite):
                suite = future_to_suite[future]
                result = future.result()
                ordered[suite] = result

        return [ordered[s] for s in suites]

    # ── Internals ─────────────────────────────────────────────────────────────

    def _run_and_report(self, suite: str) -> ContainerResult:
        name = Path(suite).stem
        print(f"  ▶ [{name}] starting container ...")
        result = self.run_suite(suite)
        icon = "✅" if result.passed else "❌"
        rate = f"{result.suite_pass_rate:.0%}" if result.results else "—"
        print(f"  {icon} [{name}] done in {result.duration_s:.1f}s  pass_rate={rate}")
        if result.error:
            print(f"     ⚠ error: {result.error}")
        return result

    @staticmethod
    def _read_results_file(path: Path) -> tuple[list[dict], Optional[str]]:
        """Read eval_results.json written by run_evals.py into the mounted volume."""
        if not path.exists():
            return [], f"results file not found at {path}"
        try:
            data = json.loads(path.read_text())
            if isinstance(data, list):
                return data, None
            if isinstance(data, dict) and "results" in data:
                return data["results"], None
            return [], f"unexpected JSON shape in {path}"
        except json.JSONDecodeError as exc:
            return [], f"JSON parse error: {exc}"


# ── Reporting ─────────────────────────────────────────────────────────────────

def print_summary(results: list[ContainerResult], min_pass_rate: float = 0.0) -> bool:
    """Print aggregated summary. Returns True if CI gate passes."""
    print(f"\n{'='*65}")
    print(f"{'Phase 6 — Containerized Test Runner Results':^65}")
    print(f"{'='*65}")

    all_cases = []
    for r in results:
        all_cases.extend(r.results)

    total_suites = len(results)
    failed_suites = sum(1 for r in results if not r.passed)
    total_cases = len(all_cases)
    passed_cases = sum(1 for c in all_cases if c.get("passed"))
    overall_rate = passed_cases / total_cases if total_cases else 0.0

    print(f"\n  Suites run  : {total_suites}   (failed containers: {failed_suites})")
    print(f"  Test cases  : {total_cases}   passed: {passed_cases}   failed: {total_cases - passed_cases}")
    print(f"  Overall rate: {overall_rate:.0%}")
    print()

    for r in results:
        icon = "✅" if r.passed else "❌"
        rate_str = f"{r.suite_pass_rate:.0%}" if r.results else "no results"
        print(f"  {icon}  {Path(r.suite).stem:<35} {rate_str}  ({r.duration_s:.1f}s)")
        if r.error:
            print(f"       ↳ error: {r.error}")

    print(f"\n{'='*65}")

    gate_ok = overall_rate >= min_pass_rate and failed_suites == 0
    if min_pass_rate > 0.0:
        status = "✓ PASS" if gate_ok else "✗ FAIL"
        print(f"  CI gate ({min_pass_rate:.0%} required): {status}")
    return gate_ok


def write_report(results: list[ContainerResult], output_path: str) -> None:
    """Write aggregated results JSON for downstream CI steps."""
    all_cases = []
    for r in results:
        for case in r.results:
            case["_suite"] = r.suite
            all_cases.append(case)

    report = {
        "phase": 6,
        "runner": "containerized",
        "suites": [
            {
                "suite": r.suite,
                "passed": r.passed,
                "pass_rate": r.suite_pass_rate,
                "duration_s": round(r.duration_s, 2),
                "case_count": len(r.results),
                "error": r.error,
            }
            for r in results
        ],
        "results": all_cases,
        "summary": {
            "total_cases": len(all_cases),
            "passed_cases": sum(1 for c in all_cases if c.get("passed")),
            "failed_containers": sum(1 for r in results if not r.passed),
        },
    }

    Path(output_path).write_text(json.dumps(report, indent=2))
    print(f"📊 Report written: {output_path}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Phase 6: Run eval suites in isolated Docker containers"
    )
    p.add_argument("--suites", nargs="+", default=DEFAULT_SUITES,
                   help="YAML suite file(s) to run (default: 3 core suites)")
    p.add_argument("--all", action="store_true", dest="run_all",
                   help="Run all 9 suites")
    p.add_argument("--image", default=FULL_IMAGE,
                   help=f"Docker image to use (default: {FULL_IMAGE})")
    p.add_argument("--build", action="store_true",
                   help="Force rebuild the Docker image before running")
    p.add_argument("--provider", default="openai", choices=["openai", "anthropic"],
                   help="LLM provider (default: openai)")
    p.add_argument("--max-parallel", type=int, default=None,
                   help="Max concurrent containers (default: all suites)")
    p.add_argument("--min-pass-rate", type=float, default=0.0,
                   help="Exit 1 if overall pass rate below this (0.0–1.0)")
    p.add_argument("--timeout", type=int, default=300,
                   help="Seconds before a container is killed (default: 300)")
    p.add_argument("--output", default="phase6_results.json",
                   help="Output JSON path (default: phase6_results.json)")
    p.add_argument("--no-build-check", action="store_true",
                   help="Skip the ensure-image step (assumes image already exists)")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if not DOCKER_AVAILABLE:
        print("❌ docker SDK not installed. Run: pip install docker")
        sys.exit(1)

    suites = ALL_SUITES if args.run_all else args.suites

    runner = ContainerRunner(
        image=args.image,
        api_key=os.environ.get("OPENAI_API_KEY"),
        provider=args.provider,
        timeout_s=args.timeout,
    )

    print(f"\n🐳 Phase 6 — Containerized Test Runner")
    print(f"   Image   : {args.image}")
    print(f"   Provider: {args.provider}")
    print(f"   Suites  : {len(suites)}")

    # Ensure image exists
    if args.build:
        runner.build_image(nocache=True)
    elif not args.no_build_check:
        runner.ensure_image()

    # Run suites in parallel containers
    results = runner.run_parallel(suites, max_parallel=args.max_parallel)

    # Report
    gate_ok = print_summary(results, min_pass_rate=args.min_pass_rate)
    write_report(results, args.output)

    sys.exit(0 if gate_ok else 1)


if __name__ == "__main__":
    main()
