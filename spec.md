Chunked Program Analysis Workflow Spec (Ticket/Agent/LLM Agnostic)

Version: 1.0
Last updated: 2026‑01‑15
Audience: Developers implementing the orchestration core + adapters

1) Objective

Build a Python orchestration core (“Controller”) that coordinates analysis of large legacy assets (COBOL programs, copybooks, JCL, etc.) using:

A ticket/work-item system (any vendor or custom)

Multiple workers/agents (“Scribes”)

A merge agent (“Aggregator”)

A review agent (“Challenger”) that can request clarifications after the first merge

Tight per-agent context limits (e.g., 4k tokens)

Key success criteria

Works with any ticket system via adapters; no hard-coded ticket schema.

Scales to huge files via chunking + hierarchical merges.

Supports a challenger loop that can dispatch targeted follow-up work to one or more chunks.

Idempotent and resumable: safe to re-run after crashes/timeouts without duplicate work.

2) Design Principles (Normative)

Artifacts are the source of truth; tickets are pointers.
Tickets MUST NOT embed large “all summaries” objects. They should reference manifest + artifact URIs.

Deterministic chunking per artifact version.
For the same source snapshot + splitter profile, chunk boundaries and chunk_ids MUST be stable.

Bounded context per task.
No ticket may require reading more content than fits the configured context budget. Large merges MUST be hierarchical.

Structured outputs.
Chunk and merge results MUST be produced in a structured, mergeable form (JSON/YAML). Prose-only summaries are insufficient for reliable aggregation.

Explicit uncertainty.
If a worker lacks context, it MUST record “unknowns” / “open questions” rather than guessing.

Controller reconciles desired state.
The controller SHOULD operate as a reconcile loop: observe state → compute missing work → create it → gate/advance.

3) Actors and Responsibilities
3.1 Program Manager (PM)

Creates initial “root analysis requests” (one per artifact).

Provides source reference(s) pinned to a version (hash/commit).

3.2 Scribe (Chunk Analyst)

Claims chunk tickets and produces chunk result artifacts.

3.3 Aggregator (Merger)

Claims merge tickets and produces merged documentation artifacts.

3.4 Challenger (Reviewer)

Reads the final (or near-final) merged documentation.

Produces questions/issues if anything is unclear, incomplete, inconsistent, or missing (e.g., error handling, restart logic).

Proposes a resolution plan that identifies which scopes need follow-up work.

3.5 Controller (Orchestration Core)

Creates chunk and merge tickets based on a manifest (chunk plan + merge DAG).

Gates merge tickets until prerequisites complete.

After challenger issues, dispatches follow-up tickets to relevant scope(s).

Triggers patch merges and optionally reruns challenger until acceptance criteria met.

Important: The controller role can be implemented as a standalone service or as logic executed by the worker that claims specific tickets. The spec does not mandate deployment shape.

4) Canonical Entities (Implementation-Agnostic)
4.1 Artifact

A versioned input or output object stored in an artifact store.

Required fields

artifact_id (stable logical name, e.g., DRKBM100.cbl)

artifact_type (COBOL | COPYBOOK | JCL | OTHER)

artifact_version (hash/commit/content hash)

artifact_uri (how to fetch content)

metadata (optional; size, repo path, etc.)

4.2 Work Item (Ticket)

A unit of work tracked in a ticketing system.

Required fields

work_id

work_type (see §6)

status (see §5)

payload (small JSON object; MUST include references to artifacts/manifests)

parent_work_id (optional)

depends_on (optional list of work IDs; or emulated gating)

4.3 Manifest

A JSON/YAML document stored in the artifact store describing the workflow plan and relationships.

At minimum, a manifest includes

job_id (unique for the analysis run)

artifact_ref (id/type/version/uri)

analysis_profile (documentation template / extraction profile)

splitter_profile

context_budget

chunks[] (chunk specs)

merge_dag[] (merge nodes)

review_policy (challenger rules)

artifacts (where to write results)

5) Status Model (Canonical)

Ticketing systems vary. Map your internal statuses to the canonical model:

NEW – created, not eligible to claim

READY – eligible to claim

IN_PROGRESS – claimed/leased by a worker

BLOCKED – waiting for dependencies/inputs

DONE – completed successfully with valid outputs

FAILED – completed with error; may be retried

CANCELED – no longer needed (optional)

Requirements

A worker MUST NOT start work on BLOCKED items unless it can unblock them deterministically.

Merge and patch merge items MUST be BLOCKED until all required inputs are DONE.

6) Work Types (Tickets)

This spec defines canonical work types; systems may rename them. Payload schemas MUST be preserved.

6.1 DOC_REQUEST (Root)

“Generate documentation for this artifact.”

Payload MUST include

job_id

artifact_ref

analysis_profile

context_budget

splitter_profile

manifest_uri (optional initially)

Outputs

Either:

Direct final doc (if small), OR

manifest_uri (chunk plan + merge DAG) that drives the rest.

6.2 DOC_PLAN (Optional Separate Planning Work)

If planning is expensive or you want strict separation, DOC_REQUEST may create a DOC_PLAN.

Payload MUST include

job_id, artifact_ref, analysis_profile, splitter_profile, context_budget

Output

manifest_uri

If you skip DOC_PLAN, planning can occur within the handler for DOC_REQUEST. Either is acceptable; the key is that a manifest is produced.

6.3 DOC_CHUNK

“Analyze a specific chunk of the artifact.”

Payload MUST include

job_id

artifact_ref

manifest_uri

chunk_id

chunk_locator (line range OR semantic locator like division/paragraph list)

result_uri (where chunk result is written)

Output

Chunk Result Artifact (see §7.1)

6.4 DOC_MERGE

“Merge a bounded set of chunk results (or child merges) into a higher-level summary.”

Payload MUST include

job_id

artifact_ref

manifest_uri

merge_node_id

input_uris (or pointer to inputs in manifest)

output_uri

Output

Merge Result Artifact (see §7.2)

6.5 DOC_CHALLENGE

“Review the current best documentation and ask questions / raise issues.”

Payload MUST include

job_id

artifact_ref

doc_uri (the merged documentation artifact to review)

doc_model_uri (machine-readable structure with section→source traceability; see §7.3)

challenge_profile (what to look for: error handling, I/O, restartability, etc.)

output_uri (where issues are written)

Output

Challenge Result Artifact (Issue Set + Resolution Plan; see §7.4)

6.6 DOC_FOLLOWUP (Issue-driven targeted work)

“Answer a specific challenger question by analyzing a scoped subset of source or prior outputs.”

Payload MUST include

job_id

artifact_ref

issue_id

scope (one of: chunk_id(s), paragraph list, line ranges, or cross-cutting query plan)

inputs (URIs to relevant chunk results and/or source slices)

output_uri

Output

Follow-up Answer Artifact (see §7.5)

6.7 DOC_PATCH_MERGE

“Apply follow-up answers to the documentation and update the doc model.”

Payload MUST include

job_id

artifact_ref

base_doc_uri + base_doc_model_uri

inputs (follow-up answer URIs)

output_doc_uri + output_doc_model_uri

Output

Updated doc + updated doc model

6.8 DOC_FINALIZE

Optional: produce final deliverables (markdown, PDF, trace report) and mark the job accepted.

7) Output Artifact Schemas (Normative)

All outputs MUST be structured and include provenance.

7.1 Chunk Result Artifact (DOC_CHUNK output)

Minimum fields:

job_id, artifact_id, artifact_version

chunk_id, chunk_locator, chunk_kind

summary (short narrative)

facts (mergeable, machine-oriented):

symbols_defined[] (name, kind, attributes)

symbols_used[] (names)

entrypoints[] / paragraphs_defined[]

calls[] (internal paragraphs, external CALL targets)

io_operations[] (READ/WRITE/REWRITE/DELETE; file/record names if known)

error_handling[] (patterns encountered: FILE STATUS checks, ON ERROR, EVALUATE of status, ABEND, etc.)

evidence[]:

{"type":"line_range","start":X,"end":Y,"note":"..."}

or stable source references depending on your locator model

open_questions[] (explicit unknowns; what else is needed)

confidence (0..1)

7.2 Merge Result Artifact (DOC_MERGE output)

Minimum fields:

job_id, artifact_id, artifact_version

merge_node_id

coverage:

included input IDs

missing/failed inputs

consolidated_facts:

merged call graph edges

merged IO map

consolidated error handling behaviors

conflicts[]:

what disagreed, which inputs, suggested follow-up scope

narrative_sections[] (optional if doc is separate)

doc_fragment_uri (optional: a rendered fragment for this merge node)

7.3 Documentation Model Artifact (required for challenger routing)

The final “doc” should not be just markdown; you need traceability so challenger issues can be routed.

Minimum fields:

doc_uri (rendered human doc)

sections[] where each section includes:

section_id, title

content (or pointer)

source_refs[] (chunk_ids and/or evidence references used)

index (optional but recommended):

symbol → chunk_ids

paragraph → chunk_id

file → chunks that touch it

7.4 Challenge Result Artifact (DOC_CHALLENGE output)

Minimum fields:

job_id, artifact_id, artifact_version

issues[], each issue includes:

issue_id (deterministic or generated)

severity (BLOCKER|MAJOR|MINOR|QUESTION)

question / problem_statement

doc_section_refs[] (which doc section is unclear)

suspected_scopes[] (chunk_ids, paragraph names, or “unknown”)

routing_hints (symbols/paragraphs/files mentioned)

resolution_plan (see below)

resolution_plan (top-level):

list of recommended follow-up tasks (bounded scopes)

whether patch merge is required

7.5 Follow-up Answer Artifact (DOC_FOLLOWUP output)

Minimum fields:

issue_id

scope (what was analyzed)

answer (clear text)

facts (structured, mergeable)

evidence (line ranges / refs)

confidence

8) Workflow: End-to-End State Machine (Normative)

This section is the “truth table” of what the controller must do.

Phase A — Request & Plan

PM creates DOC_REQUEST for each artifact (COBOL program, copybooks, JCL).

Controller (or planning worker) produces manifest:

chunk list

merge DAG with bounded fan-in

Controller creates:

DOC_CHUNK tickets for each chunk

DOC_MERGE tickets for each merge node

Controller sets DOC_MERGE tickets to BLOCKED until prerequisites are DONE.

Phase B — Chunk Analysis

Scribes claim DOC_CHUNK tickets (READY only).

Each writes a chunk result artifact and marks ticket DONE.

Phase C — Hierarchical Merge

When all inputs for a merge node are DONE, controller transitions that merge ticket to READY.

Aggregator claims and runs merges bottom-up until the root merge produces:

doc_uri

doc_model_uri

Controller creates DOC_CHALLENGE ticket and sets it READY.

Phase D — Challenger Review

Challenger reads doc + doc model and produces a Challenge Result artifact (issues + resolution plan).

If no issues above threshold:

Controller creates DOC_FINALIZE (optional) and marks job complete.

If issues exist:

Controller creates follow-up work (Phase E).

Phase E — Follow-up Dispatch & Patch Merge

Controller converts each issue’s resolution plan into one or more DOC_FOLLOWUP tickets with bounded scopes.

Scribes answer follow-ups; each emits a follow-up answer artifact.

When required follow-ups are complete, controller creates a DOC_PATCH_MERGE ticket.

Patch merge updates doc_uri + doc_model_uri and records what changed.

Phase F — Re-challenge Loop (Optional)

Controller MAY re-run DOC_CHALLENGE on the updated doc until:

issues resolved, OR

iteration limit reached, OR

only minor issues remain (policy-driven)

9) How the Controller Routes Challenger Issues to Chunks (Normative)

Because challenger questions can be cross-cutting (e.g., “error handling”), routing MUST be deterministic and bounded.

9.1 Inputs available for routing

Controller should use, in order:

doc_model.sections[].source_refs (best)

issue.suspected_scopes (from challenger)

routing_hints (symbols/paragraphs/files)

Manifest chunk boundaries (line ranges)

Chunk result indexes (symbol → chunk, paragraph → chunk)

9.2 Routing algorithm (required behavior)

For each issue:

If suspected_scopes contains chunk IDs → create follow-ups per chunk.

Else if issue references doc sections with source refs → use those chunk IDs.

Else if issue references symbols/paragraphs → consult indexes (doc model index or consolidated facts).

Else:

create a bounded “cross-cutting follow-up plan”:

one follow-up per merge-level or per division (e.g., PROCEDURE parts)

plus an “issue-specific merge” that consolidates answers.

9.3 Scope size constraints

A follow-up MUST target one of:

1 chunk

a small list of chunks (max configurable; e.g., 3–5)

a merge node output (bounded already)

a single concern within one division (bounded by planner)

If the issue appears to require “whole program re-analysis”, controller MUST split it into bounded follow-ups (e.g., “error handling in PROCEDURE_PART1”, “…PART2”, etc.) rather than creating one giant task.

10) Chunking and Merge Planning Requirements
10.1 Chunk planner (splitter) requirements

Planner MUST:

Prefer semantic boundaries when available (COBOL divisions/sections/paragraphs).

Ensure chunks fit within context budget after overhead (prompt + required context refs).

Create chunk kinds to support targeted follow-ups later (e.g., separate DATA DIVISION and PROCEDURE parts).

10.2 Merge DAG requirements

Merge nodes MUST have bounded fan-in (suggested: 8–20 inputs).

Merge DAG MUST be a DAG (no cycles).

Root merge node produces the final doc + doc model.

Planner MUST include merge nodes for large subdivisions (e.g., PROCEDURE_PART1..N) so challenger can target those units later.

11) Iteration / Cycle Model (Recommended)

To support challenger loops cleanly, the workflow SHOULD maintain an iteration number per job:

cycle=1: initial doc

cycle=2: challenger review + follow-ups + patch merge

etc.

Artifacts should include cycle in their path/URI so you can compare versions and roll back.

Your current ticket snapshot shows a cycle_number field already being used, with cycle 1 for documentation and cycle 2 for “chrome” issue tickets. 

.war_rig_tickets

12) Idempotency and Concurrency Requirements
12.1 Ticket creation idempotency

For any planned work item, controller MUST compute a stable idempotency key from:

job_id

work_type

artifact_version

(chunk_id or merge_node_id or issue_id)

If the ticket system can’t enforce uniqueness, controller MUST maintain a manifest map idempotency_key → work_id in the artifact store and avoid creating duplicates.

12.2 Artifact write idempotency

Workers MUST write outputs to deterministic output_uri derived from the payload.
If output already exists and passes validation, worker MUST mark ticket DONE without recomputation.

12.3 Leasing / claiming

Ticket backend MUST support one of:

lease/claim semantics, OR

optimistic locking to prevent two workers doing the same ticket simultaneously.

13) Observability Requirements

Controller SHOULD emit events/logs for:

manifest creation: chunk count, merge levels

ticket creation counts by type

progress: % chunks complete, % merges complete

challenger output: issue counts by severity

follow-up dispatch: number of follow-ups per issue

patch merges: sections changed, issues addressed

14) Non-Normative Appendix: Mapping to Your Current Ticket Snapshot

Your current ticket JSON illustrates one concrete implementation with fields like ticket_id, ticket_type, state, file_name, cycle_number, parent_ticket_id, and per-chunk metadata such as division, chunk_id, start_line, end_line, plus a merge ticket and “chrome” issue tickets. 

.war_rig_tickets

A.1 Example mapping (suggested)

documentation → DOC_REQUEST

documentation_chunk → DOC_CHUNK

documentation_merge → DOC_MERGE

chrome → DOC_CHALLENGE outputs or DOC_FOLLOWUP/issue tracking (depending on how you use it)

A.2 Key change recommended by this spec

Your snapshot shows chunk tickets carrying all_chunk_summaries (and merge tickets too). 

.war_rig_tickets


This spec recommends replacing that with:

manifest_uri on each ticket, and

chunk results stored in the artifact store, referenced by URI

This keeps tickets small and prevents context blowups.

15) Acceptance Tests (What “done” looks like)

Implementations SHOULD be validated against scenarios:

Huge COBOL file

Controller creates deterministic chunks and hierarchical merges.

Final doc produced without any ticket containing large embedded summary objects.

Restart/resume

Kill workers mid-run; restart; no duplicate chunk tickets; merge resumes.

Challenger asks cross-cutting question (“Explain error handling for file reads/writes”)

Controller routes into bounded follow-ups across relevant PROCEDURE chunks.

Patch merge updates the error-handling section and doc model traceability.

Ambiguous routing

If challenger can’t identify scope, controller creates a bounded cross-cutting follow-up plan rather than one huge task.