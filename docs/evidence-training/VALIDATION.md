# Validation receipt — 2026-10-01

Source baseline: `2cdce8456617d30fa93a11f3ac36db55efd0dbd5`.
Checkout: `C:/Users/Owner1/.codex/worktrees/nlm-training-modalities/NLM`.
No dependency installation, full repository build, GPU execution or service deployment.

| Command | Result |
| --- | --- |
| `python -B -m unittest discover -s tests/evidence_training -p 'test_*.py' -v` | 39 passed in 11.043s |
| `python -B tests/evidence_training/export_fixture.py` | Created actual fixture model/checkpoint/evaluation and exact-byte hash receipt; 6 training and 4 heldout rows; confusion `[[2,0],[0,2]]` |
| `git diff --check` | Passed for tracked diff at check time; new source separately reviewed |

The 39 tests comprise 9 algorithm/manifest tests, 6 HTTP-contract fixture tests,
and 24 durable worker-fixture tests. Async unittest diagnostics occasionally
reported callbacks exceeding 0.1 seconds during temporary SQLite operations;
there were no failures. Elapsed test time is a reproduction receipt, not a CPU
throughput, p50/p95 latency, RSS, cost or approved-environment benchmark.

Engine tests cover exact output repeatability; actual changed weights and falling
training loss; heldout confusion and model reload; exact resume at every epoch
boundary; train-only scaling; malformed masks/units/time/numbers; missingness;
group/capture-hash leakage; checkpoint/config/runtime pins; and corrupt artifacts.

HTTP fixture tests cover original JWT + two scope selectors, worker-proof headers,
no identity override/fallback/redirect, HTTPS restrictions, reauthentication,
bounded hashed idempotency keys, admission-versus-verified archive state, response
scope/content identity, exact byte readback, and lease-fence mismatch. These use
a request/response transport fixture, not a deployed service.

Worker tests use temporary, explicitly test-only SQLite. They cover competing
claims in two threads/connections, restart with new worker/store instances,
simulated abrupt process loss, every selected checkpoint cursor, lease expiry,
stale tokens/fences, cancellation before and racing completion, foreign dataset
and capture references, all four scope dimensions, admission engine pin mismatch,
bad artifact upload receipts, corrupted/readback-unavailable bytes, lost upload
and completion acknowledgements, and server-side re-verification before success.
No actual service process restart, PostgreSQL or S3 test is implied by this suite.
Sibling MINDEX integration receipts must be consulted separately.

Release gates remain: real verified Supabase identity and membership/revocation,
approved scoped worker policy, exact compatible NLM source/runtime installation
in API and worker, real private archive verification, MINDEX PostgreSQL migration
and recovery rehearsal, licensed observed data with scientifically reviewed
calibration/split independence, field validation, approved capacity and benchmark
environment. No current production state or old SINE live receipt was requalified.
