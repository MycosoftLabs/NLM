# NLM — Eagle `scene_index` batch embedding pipeline (Apr 17, 2026)

## Purpose

Populate `eagle.scene_index` from clip/snapshot artifacts using **batch** NLM embedding jobs—not inline in the HTTP request path at scale.

## Data contract (MINDEX)

| Column | Role |
|--------|------|
| `video_source_id` | FK to `eagle.video_sources.id` |
| `observed_at` / `clip_start` / `clip_end` | Time window for the scene |
| `vlm_summary`, `ocr_text` | Text features for search + RAG |
| `embedding vector(768)` (or agreed dim) | Semantic retrieval; must match NLM export dim |

## Pipeline (recommended)

1. **Inputs:** Headless snapshot service or approved clip store writes metadata + object path to a queue or staging table.
2. **NLM batch job:** Read pending rows → run encoder → `UPDATE eagle.scene_index SET embedding = …` (or insert new rows).
3. **Scheduling:** MAS cron/n8n or MINDEX worker; rate-limit to GPU/CPU budget.
4. **Privacy:** Set `privacy_class` on parent source; unified-search / Fluid Search must not expose `operator_only` / `restricted` rows to public tiers (enforce in SQL or post-filter).

## Verification

- Row counts in `eagle.scene_index` increase after batch runs.
- MINDEX `GET .../unified-search?types=eagle_video` returns sources; when scene text is indexed, extend search to join `scene_index` as needed (separate task).
