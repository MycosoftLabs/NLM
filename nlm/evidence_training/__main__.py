"""Explicit single-job CLI. No polling daemon, deployment, or compute launch."""

import argparse
import asyncio
import json
import os
import sys

from .engine import engine_pin
from .http_port import MindexHttpPort
from .worker import run_job


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("engine-pin", help="Print engine source/license/runtime pins without training")
    command = commands.add_parser("worker", help="Run one already admitted authorized MINDEX job")
    command.add_argument("--base-url", required=True, help="Approved MINDEX HTTPS base including /api/mindex")
    command.add_argument("--tenant", required=True)
    command.add_argument("--project", required=True)
    command.add_argument("--job", required=True)
    command.add_argument("--worker-id", required=True)
    command.add_argument("--lease-seconds", type=int, default=120)
    args = parser.parse_args()
    if args.command == "engine-pin":
        print(json.dumps(engine_pin(), indent=2))
        return 0

    def worker_headers():
        token = os.environ.get("NLM_WORKER_AUTHORIZATION", "")
        return {"X-NLM-Worker-Authorization": token, "X-NLM-Worker-Id": args.worker_id} if token else {"X-NLM-Worker-Id": args.worker_id}

    try:
        port = MindexHttpPort(args.base_url, lambda: os.environ.get("NLM_ORIGINAL_USER_JWT", ""),
                              args.tenant, args.project, worker_headers=worker_headers)
        result = asyncio.run(run_job(port, args.job, args.worker_id, lease_seconds=args.lease_seconds))
        print(json.dumps(result, sort_keys=True))
        return 0
    except Exception as exc:
        # Do not log response bodies, JWTs, authorization headers or private captures.
        print(json.dumps({"state": "unavailable", "error_type": type(exc).__name__}), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
