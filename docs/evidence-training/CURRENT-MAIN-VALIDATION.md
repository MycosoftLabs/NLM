# Brief 05 current-main training candidate

Base: NLM `8264fbfa4d4b6f8fefc9ec5890b50a01567b31d1` (2026-10-03).
Accepted source: NLM `1b0a0237150ab73b570c21dc9b14b9ee1103803f`.

This candidate adds the bounded `thermal-softmax-v1` evidence trainer, HTTP port, worker contracts, authored synthetic capture manifest, and offline fixture tests. It does not bind the full NLM model, GPU training, licensed field data, or production services.

The engine source and manifest bind to the unchanged current-main `LICENSE` SHA-256 (LF-normalized) `d1a5692f948e2e6b44693d6c84382c92264eac72c53e2d8d367d7af47fee1f6f`; the root license file was not changed.

Local validation on this current-main derivative: `python -B -m unittest discover -s tests/evidence_training -p 'test_*.py' -v` ran 39 tests and passed. It used the existing Python 3.12 interpreter and its installed `httpx`; no dependencies were installed. Worker persistence in the tests uses temporary SQLite fakes. No service, PostgreSQL, S3, GPU, browser, or model deployment was started.
