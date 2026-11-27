# AWS infrastructure reference and legacy file register

Verified 2026-11-27; account 376129856177, us-east-1. Production servers are AWS-only. This source branch is not deployed. Canonical system map: MycosoftLabs/mycosoft-mas root `MYCOSOFT_AWS_INFRA_CURRENT.md`.

Source base: `8264fbfa4d4b6f8fefc9ec5890b50a01567b31d1`. Supplied Grok patches are rebased by exact hunk context/three-way merge; unavailable MycoBrain base ef93c57 was reconciled against current main using complete matching context. Historical code bodies remain present.

## Configuration

MINDEX: `http://mindex-internal.mycosoft.com:8000`; MAS: `http://mas-internal.mycosoft.com:8001`; MycoBrain: `http://mycobrain-internal.mycosoft.com:8003`. Mycorrhizae uses `http://internal-mycosoft-core-int-21300503.us-east-1.elb.amazonaws.com:8002`; n8n MAS/MYCA use the same internal ALB ports5678/5679. Website is `https://mycosoft.com`; Origin is `https://origin-aws.mycosoft.com:8443`. These private backends are server-side only. Never put service secrets into NEXT_PUBLIC variables. Database access uses Secrets Manager application credentials and the TLS-verified RDS Proxy; see the canonical map.

Unknown optional LAN/hypervisor/device integrations need explicit configuration and separate verification; a retained literal in those bodies is not a discovered AWS replacement. Tests, source datasets, migrations and signed records retain evidence. Retired CLI guards only stop the standalone old SSH/Proxmox utility; importable library bodies remain. AWS-aware utilities were preserved. Production AWS services were not restarted or redeployed.

## GitHub workflow inventory

No tracked GitHub Actions workflows found at this source base.

## Complete matching-file register

Rows are a reviewed inventory, not evidence that an optional connector is used in production. Each replacement either points to the verified AWS service, retires the local topology instruction, or preserves a non-current evidence/configuration record.

| File | Disposition | Replacement / retained-reference meaning |
|---|---|---|
| `nlm/client.py` | AWS / AWS_ENDPOINTS_CORRECTED | http://192.168.0.189:8000 -> http://mindex-internal.mycosoft.com:8000 |
| `nlm/training/dataset.py` | AWS / AWS_ENDPOINTS_CORRECTED | http://192.168.0.189:8000 -> http://mindex-internal.mycosoft.com:8000 |
