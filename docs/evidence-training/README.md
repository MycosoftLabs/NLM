# Thermal feature training evidence

Engineering brief 05, October 1, 2026. This package implements one bounded trainer,
`thermal-softmax-v1`, over declared temperature and temperature-gradient features.
It is a real supervised softmax fit using CPU Python float64 arithmetic. Its local
fixtures do not qualify sensors, field performance, deployed identity, private S3
retention, a production worker, or the complete Nature Learning Model.

## Source and scope

- Repository: `https://github.com/MycosoftLabs/NLM.git`.
- Baseline: `2cdce8456617d30fa93a11f3ac36db55efd0dbd5`.
- Branch: `codex/nlm-evidence-training`.
- Implementation: `nlm/evidence_training/**`; tests: `tests/evidence_training/**`.
- Original NLM source and all other application paths are unchanged.
- The repository `LICENSE` is proprietary. Existing `pyproject.toml` says MIT;
  that pre-existing metadata conflict does not grant a new license. The engine
  records the reviewed root LICENSE SHA256 with CRLF normalized to LF. Invented
  fixture captures are authored test inputs under the repository terms; no
  external dataset, pretrained model, or third-party model asset is included.

The full source model in `nlm/model/nlm_model.py` constructs spatial, temporal,
spectral/sensory, world-state, self-state and action/intent encoders, then performs
sparse fusion, `TemporalSSMStack`, and prediction heads. Its forward call uses a
single-step sequence (`hidden.unsqueeze(1)`); `ssm_blocks.py` initializes scan
state for each call. The thermal feature classifier does not instantiate these
encoders, train persistent temporal state, or imply full NLM qualification.
Other fourteen modality engines remain unbound.

## Immutable data and scientific boundaries

`contracts.validate_manifest(raw_bytes)` accepts exactly the shared
`mycosoft.nlm.training-manifest.v1` envelope with a thermal
`mycosoft.nlm.modality-dataset.v1` dataset, presence masks, retained capture refs,
an extractor pin, and timestamp semantics. The authoritative manifest identifier
is SHA256 of the exact archived UTF-8 JSON bytes. Do not hash a reserialized
JavaScript or Python object to identify the same archive. Duplicate JSON keys,
nonfinite numbers and lone surrogate text are rejected. Engine-owned artifact
JSON is deterministic Python JSON, not an implementation of RFC 8785.

The engine requires channel order `temperature, gradient` with units `degC,
degC_per_m`. The extractor pin hashes the exact literal UTF-8 schema bytes in
`FEATURE_SCHEMA_BYTES`; it identifies a supplied-feature contract, not a raw
capture decoder. Every row links to exactly one original capture. Split/group
provenance, raw artifact hash/length, source class, license, calibration, original
timestamps, notes and masks remain in immutable manifest bytes. A capture or an
identical capture hash cannot occur in both training and holdout, even under
renamed groups. Independence and license/calibration declarations remain operator
evidence; the code cannot establish physical independence or legal sufficiency.

Missing values retain `null` with a false presence mask. Admission for this engine
rejects incomplete labels/features; zero remains an observed number. Means and
population standard deviations use training rows only; a zero variance feature
uses scale 1. The classifier uses fixed row order, stable softmax, full-batch
gradient descent, L2 on feature weights only, and explicitly specified seeded LCG
initialization. No randomness occurs after initialization. Checkpoints contain
epoch cursor, all weights, exact config, training scaler, complete loss history,
label map, split identities, manifest hash, and engine/runtime/license pins.
Epoch-boundary resume is byte-identical on the pinned runtime. No tolerance or
reproducibility is claimed across differing runtime pins.

Bounds: 8 MiB manifest and each capture/artifact; at most 20 captures, 2,000 rows,
32 classes, 2 channels, 200 epochs; uint32 seed; learning rate `(0,1]`; L2 `[0,1]`.
Feature magnitude is bounded by `1e6`, temperature below -273.15 degC is rejected,
and standardized values/logits must remain finite. Every class requires two
training rows and a held-out row. The heldout report includes actual confusion
counts, predictions, row counts, dataset/source scope and limitations; softmax
scores are not calibrated confidence or general field accuracy.

## UI, admission, worker, artifact readback

1. The website retains original captures and exact manifest bytes through brief09
   private retention; user scope comes from verified Supabase identity plus
   server-owned membership, never a manifest owner field.
2. MINDEX `nlm-training.v1` admission validates the manifest/captures and stores
   immutable configuration and a server-derived `engine_pin`. Admission is not
   running or success. The API and worker must run matching engine/runtime pins.
3. `worker.run_job` claims the canonical job through the MINDEX port, verifies
   the engine pin, reads every capture and manifest with exact hash/length checks,
   and fits at most one epoch between lease checks.
4. Every accepted checkpoint and terminal mutation carries token + monotonically
   increasing lease epoch. The MINDEX PostgreSQL implementation owns atomic
   current-state, expiry, fence and cancellation checks. Worker crashes leave a
   lease that can expire and resume from the last accepted immutable checkpoint.
5. Outputs are retained by shared09, read back byte-for-byte, and the retained
   model reproduces heldout predictions before completion. MINDEX independently
   validates model/evaluation/checkpoint consistency with the admitted manifest.
   A status string or successful upload admission cannot produce success.

The real `MindexHttpPort` uses `/api/mindex/nlm-training/v1` and
`/api/mindex/retention/v1`. It forwards original bearer JWT and tenant/project
selectors on each request, rejects redirects, permits remote HTTPS only, bounds
request/response sizes and waits, checks scoped archive receipts, and performs
exact content readback. Extra worker headers can carry a separately authorized
worker proof; the server's independently injected worker-authorizer is required.
No default worker policy, user impersonation, service-key identity fallback, or
production store is implemented here. An expiry/revocation stops subsequent
requests. Configured endpoint/identity/worker policy and matching source/runtime
pins are release gates, not a deployment performed by this change.

The port may upload an object just before losing its compute lease. Such an
object remains private shared retention evidence and cannot advance the job
without the current fence. No cleanup or automatic deletion is attempted.
Transient retention outages preserve resumability; permanent malformed training
evidence is failed only through the canonical fenced job operation.

## Reproduction

From the isolated NLM checkout, using existing Python 3.12 (no install):

```powershell
python -B -m unittest discover -s tests/evidence_training -p 'test_*.py' -v
python -B tests/evidence_training/export_fixture.py
python -B -m nlm.evidence_training engine-pin
git diff --check
```

`fixture-receipts/` contains actual local training outputs and original synthetic
capture bytes. `receipt.json` binds all files by exact SHA256 and byte length. The
fixture has six training rows, four heldout rows, two classes and four declared
capture groups. The resulting confusion matrix is `[[2,0],[0,2]]`. This tiny
invented separable example checks mechanics only and is not scientific accuracy.

Worker tests use a explicitly marked **test-only SQLite fixture**, temporary
files and fault injection. It demonstrates process-instance restart, atomic
competing lease checks and cancellation races locally; it is not a production
MINDEX adapter and does not replace the sibling PostgreSQL tests. HTTP tests use
a bounded request/response fixture; real Supabase and private archive readback
remain separate qualification gates.

After separate approval and nonproduction provisioning, a reviewed operator can
run one already admitted job explicitly (never from an automatic deployment):

```powershell
# Supply original user JWT through NLM_ORIGINAL_USER_JWT in the authorized host.
# Supply separately authorized worker proof via NLM_WORKER_AUTHORIZATION if policy uses it.
python -B -m nlm.evidence_training worker --base-url https://APPROVED_HOST/api/mindex --tenant TENANT_UUID --project PROJECT_UUID --job JOB_UUID --worker-id approved-worker
```

No credential value belongs in commands, docs, receipts, fixtures, or Git.

## Qualification and release

Local algorithm/fixtures, retained local review files, and HTTP serialization are
implemented. Deployed authentication/membership, private encrypted/versioned
archive, PostgreSQL migration/restore rehearsal, independently authorized worker
policy, actual sensor inputs/calibration/licensing review, scientific external
validation and benchmark environment remain required. CPU RSS, GPU memory,
wall-time p50/p95, throughput and cost are **not measured qualification results**.
The host had approximately 2.08 GiB free of 31.70 GiB during the read-only audit;
that is a transient observation, not a deployment sizing result. No local torch
distribution was found in Python312 site-packages; this engine requires none.
Existing SINE CNN/TorchScript infrastructure remains a future separate qualified
trainer: its ESC-50 catalog records CC BY-NC 3.0 and is not treated as a general
commercial dataset grant.

No migrations, service starts, GPU/cloud runs, paid compute, deployments or model
promotions were performed. MINDEX owns its migration and reversal plan. For a
manual blue/green release, pin this exact package and runtime, preserve previous
source/artifacts, run cross-tenant/expiry/fence/cancellation/archive-readback
acceptance tests on approved nonproduction infrastructure, then review an
explicit traffic switch and rollback. Changing engine source or runtime requires
new admission; old checkpoints are intentionally refused by a mismatched engine.
Rollback does not erase retained evidence or undo database changes. Do not push
until workflows and external deploy hooks are proven unable to auto-deploy.
