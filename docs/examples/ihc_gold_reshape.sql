-- IHC gold reshape — example per-batch sidecar for the PathKidneyIhc workflow.
--
-- HOW IT RUNS
--   Drop this beside a review batch in the inbox as
--       inbox/fc_reviews/<batch>/<batch>.sql
--   `oncai ingest fc_reviews` mirrors it to lake/fc_reviews/<batch>.sql, and
--   `oncai build-db` runs it immediately after building that batch's reviewed
--   table extractions_silver."<batch>". So the script is *aligned with one
--   batch* and refers to that batch's silver table by name.
--
--   Replace every `kidney_v1` below with your batch name (the silver table /
--   the .sql file's stem). These statements CREATE OR REPLACE, so each batch
--   owns its own gold tables; if you'd rather pool batches into one shared
--   table, prefix the names (ihc_results_kidney_v1) or INSERT into a table you
--   created once.
--
-- WHY IT'S NEEDED
--   The silver table is event-grain: PathKidneyIhc registers two event tools
--   (record_ihc_result, flag_report_for_review) that share the one wide table,
--   so a flag row is null across every IHC-result column and vice-versa. Gold
--   is where we split the event types apart and pivot the markers into a dense,
--   analysis-ready shape. Only approved / auto-accepted events reach silver, so
--   there is no verdict to filter on here.

-- 1. One tidy row per reviewed IHC / FISH result ---------------------------
--    The analytic fields, plus enough provenance to audit each value back to
--    the reviewed event it came from.
CREATE OR REPLACE TABLE extractions_gold.ihc_results AS
SELECT
    mrn,
    note_id,
    note_date,
    specimen_id,
    given_test_name,
    standardized_test_name,
    given_result,
    standardized_test_status,
    standardized_test_intensity,
    standardized_test_extent,
    standardized_test_pattern,
    flag_for_sub_specimen_heterogeneity,
    -- provenance back to the reviewed event
    acceptance_reason,
    review_verdict,
    reviewer,
    reviewed_at,
    batch_name,
    event_key
FROM extractions_silver."kidney_v1"
WHERE event_type = 'record_ihc_result'
  AND standardized_test_name IS NOT NULL
ORDER BY mrn, note_id, specimen_id, standardized_test_name;

-- 2. Reports a human flagged as too ambiguous for the toolset --------------
CREATE OR REPLACE TABLE extractions_gold.ihc_review_flags AS
SELECT
    mrn,
    note_id,
    note_date,
    reason,
    review_comment,
    reviewer,
    reviewed_at,
    batch_name,
    event_key
FROM extractions_silver."kidney_v1"
WHERE event_type = 'flag_report_for_review'
ORDER BY mrn, note_id;

-- 3. Per-specimen marker matrix --------------------------------------------
--    One row per (patient, note, specimen); one column per canonical marker;
--    value = standardized result. This is the shape an analyst actually wants —
--    no null-by-construction columns, no JSON blobs, a marker either fired for a
--    specimen or it didn't. Heterogeneous specimens (the same marker recorded
--    twice with conflicting results) collapse to first() here and stay flagged
--    in ihc_results.flag_for_sub_specimen_heterogeneity for follow-up.
CREATE OR REPLACE TABLE extractions_gold.ihc_marker_matrix AS
PIVOT extractions_gold.ihc_results
ON standardized_test_name
USING first(standardized_test_status)
GROUP BY mrn, note_id, note_date, specimen_id;
