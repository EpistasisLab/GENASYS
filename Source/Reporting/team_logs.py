"""The files one team evaluation leaves behind, and its line in the console.

  {run_dir}/Team_Results/Team_{id}/Eval_{i}/
      team.txt                  the roster, written before the run
      run_history.log           every message of the attempt that counted, each worker
                                handoff's transcript in place, then a RUN SUMMARY
      {role}_transcripts.json   one per worker on the team: every handoff to it, with
                                its task, the input its model was shown, and each
                                message it added (tool calls with their arguments,
                                tool results, usage); "handoffs" is empty if never called
      feature_decisions.json    every accepted / rejected / pending feature, with its
                                rationale, verdict and reviews
      feature_mean_details.csv  one row per encoded column (scored evaluations only),
                                with its importance averaged over folds
      feature_fold{k}_details.csv
                                the same rows with fold k's importance, k = 1..cv_folds
      Provider_Failures/Attempt_{k}/run_history.log, {role}_transcripts.json
                                each attempt lost to the model provider, if any
  {run_dir}/team_evaluations.csv   one row per evaluation, across the run

Evaluation/result.py computes every number; this module only formats and writes it.
Errors are not caught: a log that cannot be written is a bug, and stops the run.
"""

import csv
import json
import os
from dataclasses import asdict
from typing import Any, Dict, List, Mapping, Sequence

from langchain_core.messages import ToolMessage, message_to_dict
from typeguard import typechecked

from ..Evaluation.outcome import RunOutcome
from ..Evaluation.result import FeatureColumn, EvalResult
from ..Evaluation.run_graph import GraphRun, Handoff
from ..Evaluation.scoring import FOLD_METRICS
from ..Optimizer.objectives import PERFORMANCE_LABEL
from ..Team_Builder.team import Team

TEAM_RESULTS_DIR = "Team_Results"
TEAM_EVALUATIONS_CSV = "team_evaluations.csv"
PROVIDER_FAILURES_DIR = "Provider_Failures"

_RULE = "=" * 32

# Metric columns of team_evaluations.csv: every per-fold mean, the spread of the two
# ranking metrics, and the fold and row counts. Blank when the evaluation was not scored:
# no model was trained, and a 0 would read as a measurement.
_METRIC_COLUMNS = (*FOLD_METRICS, "roc_auc_std", "pr_auc_std", "n_folds", "n_rows")
_COUNT_METRICS = ("n_folds", "n_rows")

TEAM_EVALUATION_COLUMNS = (
    "generation", "team_id", "eval_index", "attempts", "status", "performance",
    *_METRIC_COLUMNS,
    "input_tokens", "output_tokens", "cached_input_tokens", "token_count", "cost_usd",
    "eval_seconds", "eval_minutes",
    "n_features", "n_accepted", "n_rejected", "n_pending", "feature_names", "run_status")

FEATURE_DETAIL_COLUMNS = ("generation", "team_id", "eval_index",
                          *FeatureColumn.__dataclass_fields__)


def eval_dir(run_dir: str, team_id: int, eval_index: int) -> str:
    """``{run_dir}/Team_Results/Team_{id}/Eval_{i}``."""
    return os.path.join(run_dir, TEAM_RESULTS_DIR, f"Team_{team_id}", f"Eval_{eval_index}")


def attempt_dir(out_dir: str, attempt: int) -> str:
    """Where an attempt lost to the provider keeps its log, inside its ``Eval_{i}``."""
    return os.path.join(out_dir, PROVIDER_FAILURES_DIR, f"Attempt_{attempt}")


# ------------------------------------ team.txt ---------------------------------------------

@typechecked
def write_team_roster(path: str, team: Team, team_id: int, recursion_limit: int) -> None:
    """The team as evaluated: lineage, then the Supervisor and each worker's blueprint."""
    roster = team.get_roster()
    workers = sorted(role for role in roster if role != "supervisor")
    parent0, parent1 = team.get_parent_ids()
    lines = [f"{_RULE} TEAM {team_id} {_RULE}",
             f"generation: {team.get_generation()}",
             f"variation: {team.get_variation()}",
             f"parents: ({parent0}, {parent1})",
             f"recursion_limit: {recursion_limit}",
             f"team size: {len(roster)} (1 supervisor + {len(workers)} "
             f"worker{'' if len(workers) == 1 else 's'})",
             f"supervisor delegates to: {', '.join(workers) if workers else '(none: single agent)'}"]
    for role in ["supervisor", *workers]:
        blueprint = roster[role]
        lines += ["", "-" * 79, role,
                  f"  model:      {blueprint.get_model()}",
                  f"  parameters: {json.dumps(blueprint.get_params(), sort_keys=True)}"]
    _write_text(path, "\n".join(lines) + "\n")


# --------------------------------- run_history.log -----------------------------------------

def _messages(run: GraphRun) -> List[str]:
    """The team's messages, each worker handoff's transcript right after the handoff
    call that started it. A worker's final message is also on the team's state; it is
    shown once, in its transcript."""
    handoffs = {h.tool_call_id: h for h in run.handoffs}
    in_transcripts = {m.id for h in run.handoffs for m in h.messages}
    placed = set()
    blocks = []
    for message in run.state["messages"]:
        if message.id in in_transcripts:
            continue
        blocks.append(message.pretty_repr(html=False))
        if isinstance(message, ToolMessage) and message.tool_call_id in handoffs:
            blocks.append(_transcript(handoffs[message.tool_call_id]))
            placed.add(message.tool_call_id)
    missing = sorted(set(handoffs) - placed)
    # A handoff whose tool result never reached the team's state has no anchor: a run that
    # raised mid-handoff (e.g. an Ollama 500 inside a worker), but also a clean run whose
    # Supervisor ended on an empty turn right after a handoff (seen with qwen3.6, qwen3.8
    # and the mixed run). Show it at the end: a log writer must never crash the EA.
    blocks.extend(f"(handoff interrupted before reaching the team's messages)\n"
                  f"{_transcript(handoffs[call_id])}" for call_id in missing)
    return blocks


def _transcript(handoff: Handoff) -> str:
    """One handoff's messages, indented between a header and a footer."""
    title = (f"{handoff.agent} transcript: handoff {handoff.index} "
             f"(call {handoff.call_number} of {handoff.agent})")
    status = ("completed" if handoff.completed
              else "INCOMPLETE: the run stopped during this handoff")
    shown = f"What {handoff.agent}'s model was given:\n{handoff.worker_input}"
    body = "\n\n".join([shown, *(m.pretty_repr(html=False) for m in handoff.messages)])
    return "\n".join([f"    {'-' * 16} {title} {'-' * 16}",
                      *("    " + line if line else "" for line in body.splitlines()),
                      f"    {'-' * 16} end of {handoff.agent} handoff {handoff.index}: "
                      f"{status} {'-' * 16}"])


def _outcome_lines(outcome: RunOutcome) -> List[str]:
    return [f"status: {outcome.status.value} ({outcome.detail})",
            f"accepted: {outcome.n_accepted}  rejected: {outcome.n_rejected}  "
            f"pending: {outcome.n_pending}  committed: {len(outcome.features)}"]


@typechecked
def write_failed_attempt(out_dir: str, run: GraphRun, outcome: RunOutcome, attempt: int,
                         max_attempts: int) -> None:
    """The transcript of an attempt lost to the provider, and why it was lost."""
    os.makedirs(out_dir, exist_ok=True)
    summary = [f"\n{_RULE} PROVIDER FAILURE {_RULE}",
               f"attempt: {attempt} of {max_attempts} (retried)",
               *_outcome_lines(outcome), "", str(run.traceback)]
    _write_text(os.path.join(out_dir, "run_history.log"),
                "\n\n".join(_messages(run)) + "\n" + "\n".join(summary) + "\n")


# ------------------------------ {role}_transcripts.json -------------------------------------

def transcripts_file(role: str) -> str:
    return f"{role}_transcripts.json"


@typechecked
def write_worker_transcripts(out_dir: str, run: GraphRun, workers: Sequence[str],
                             header: Mapping[str, Any]) -> None:
    """``{role}_transcripts.json`` for every worker on the team, called or not.

    Each message is LangChain's own serialization (``message_to_dict``), so nothing is
    dropped and ``messages_from_dict`` reads it back. ``handoff_index`` orders handoffs
    across the files; ``run_history.log`` shows them interleaved.
    """
    strangers = sorted({h.agent for h in run.handoffs} - set(workers))
    assert not strangers, f"handoffs to {strangers}, which are not on the team {sorted(workers)}"
    for role in workers:
        handoffs = [{"handoff_index": h.index, "call_number": h.call_number,
                     "handoff_tool_call_id": h.tool_call_id, "task": h.task,
                     "supervisor_message": h.supervisor_message,
                     "worker_input": h.worker_input, "completed": h.completed,
                     "messages": [message_to_dict(m) for m in h.messages]}
                    for h in run.handoffs if h.agent == role]
        payload = {"role": role, **header, "handoffs": handoffs}
        _write_text(os.path.join(out_dir, transcripts_file(role)),
                    json.dumps(payload, indent=2) + "\n")


@typechecked
def write_run_history(path: str, run: GraphRun, result: EvalResult) -> None:
    """Every message of the attempt that counted, then its RUN SUMMARY."""
    r = result
    usage = r.cost.usage
    summary = [f"\n{_RULE} RUN SUMMARY {_RULE}",
               f"team: {r.team_id}  generation: {r.generation}  eval: {r.eval_index}",
               f"attempts: {r.attempts}"]
    summary += [f"  attempt {k} lost to the provider: {error}"
                for k, error in enumerate(r.provider_failures, start=1)]
    summary += _outcome_lines(r.outcome)
    summary.append(f"performance ({PERFORMANCE_LABEL}, mean over CV folds of the dev split): "
                   + (_performance(r) if r.score is not None else "0 (not scored)"))
    summary += [f"eval_seconds: {r.eval_seconds:.2f}",
                f"tokens: {usage.total_tokens:,} (input {usage.input_tokens:,}, output "
                f"{usage.output_tokens:,}, cached input {usage.cached_input_tokens:,})",
                f"cost_usd: {r.cost.total_usd:.6f}"]
    for role, agent in r.cost.per_agent.items():
        summary.append(f"  {role} ({agent.model}): {agent.usage.total_tokens:,} tokens, "
                       f"${agent.cost_usd:.6f}")
    if r.score is not None:
        summary += ["metrics: " + json.dumps(r.score.metrics, sort_keys=True)]
    summary += ["", f"feature decisions ({len(r.outcome.decisions)}), "
                    f"also in feature_decisions.json:",
                json.dumps(_decisions(r), indent=2)]
    if r.traceback is not None:
        summary += ["", "traceback:", r.traceback]
    _write_text(path, "\n\n".join(_messages(run)) + "\n" + "\n".join(summary) + "\n")


# ------------------------------ feature_decisions.json --------------------------------------

def _decisions(result: EvalResult) -> List[Dict[str, Any]]:
    return [{"decision": d.decision, "committed": d.committed, **asdict(d.record)}
            for d in result.outcome.decisions]


@typechecked
def write_feature_decisions(path: str, state: Mapping[str, Any], result: EvalResult) -> None:
    """Every feature the team decided on or left pending, with all its text.

    A feature rejected and later accepted appears once per round. ``reviews`` maps each
    of the team's review fields to its reviews, oldest first.
    """
    payload = {"team_id": result.team_id, "generation": result.generation,
               "eval_index": result.eval_index, "status": result.outcome.status.value,
               "final_feature_set": state["final_feature_set"],
               "decisions": _decisions(result),
               "rejected_concepts": state["rejected_concepts"]}
    _write_text(path, json.dumps(payload, indent=2) + "\n")


# ------------------------------ feature_*_details.csv --------------------------------------

@typechecked
def write_feature_details(out_dir: str, result: EvalResult) -> None:
    """The mean importance across folds, then each fold's, one row per encoded column."""
    assert result.columns, "only a scored evaluation has feature details"
    tables = {"feature_mean_details.csv": result.columns,
              **{f"feature_fold{fold}_details.csv": columns
                 for fold, columns in enumerate(result.fold_columns, start=1)}}
    for name, columns in tables.items():
        rows = [[result.generation, result.team_id, result.eval_index,
                 *[_cell(value) for value in asdict(column).values()]]
                for column in columns]
        _write_rows(os.path.join(out_dir, name), FEATURE_DETAIL_COLUMNS, rows, append=False)


# ------------------------------ team_evaluations.csv ---------------------------------------

@typechecked
def append_team_evaluation(csv_path: str, result: EvalResult) -> None:
    """Append ``result`` as one row; the header is written when the file is created."""
    r = result
    metrics = r.score.metrics if r.score is not None else {}
    usage = r.cost.usage
    names = [f.feature_name for f in r.outcome.features]
    row = [r.generation, r.team_id, r.eval_index, r.attempts, r.outcome.status.value,
           f"{r.performance:.6f}",
           *[_metric(metrics, key) for key in _METRIC_COLUMNS],
           usage.input_tokens, usage.output_tokens, usage.cached_input_tokens,
           usage.total_tokens, f"{r.cost.total_usd:.6f}",
           f"{r.eval_seconds:.2f}", f"{r.eval_seconds / 60.0:.2f}",
           len(names), r.outcome.n_accepted, r.outcome.n_rejected, r.outcome.n_pending,
           # "; " rather than ",": catalog names contain commas and " | ".
           "; ".join(names), r.outcome.detail]
    _write_rows(csv_path, TEAM_EVALUATION_COLUMNS, [row], append=True)


def _metric(metrics: Mapping[str, Any], key: str) -> str:
    if not metrics:
        return ""
    value = metrics[key]
    return str(int(value)) if key in _COUNT_METRICS else f"{float(value):.6f}"


# ------------------------------------ console ----------------------------------------------

def _performance(result: EvalResult) -> str:
    metrics = result.score.metrics
    return (f"{metrics['roc_auc']:.4f} +/-{metrics['roc_auc_std']:.4f} over "
            f"{int(metrics['n_folds'])} folds (no-skill 0.5000)  |  PR-AUC "
            f"{metrics['pr_auc']:.4f} (no-skill {metrics['positive_rate']:.4f})")


@typechecked
def print_evaluation(result: EvalResult) -> None:
    """One line per evaluation: how it ended, its score, and what it spent."""
    r = result
    score = (f"{PERFORMANCE_LABEL} {_performance(r)}" if r.score is not None
             else f"{PERFORMANCE_LABEL} 0 ({r.outcome.detail})")
    retries = f"  |  {r.attempts} attempts" if r.attempts > 1 else ""
    print(f"    Team {r.team_id} eval {r.eval_index}: {r.outcome.status.value}, "
          f"{len(r.outcome.features)} features  |  {score}  |  "
          f"{r.cost.usage.total_tokens:,} tokens, ${r.cost.total_usd:.4f}  |  "
          f"{r.eval_seconds:.1f}s{retries}", flush=True)


# ------------------------------------ helpers ----------------------------------------------

def _cell(value: Any) -> Any:
    """A CSV cell: None is blank, floats get 6 places, booleans are 0/1."""
    if value is None:
        return ""
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, float):
        return f"{value:.6f}"
    return value


def _write_rows(path: str, header: Sequence[str], rows: List[List[Any]], append: bool) -> None:
    """Write ``rows`` under ``header``; when appending, the header only for a new file.

    The csv module quotes any cell holding a comma, quote or newline, so free text
    round-trips through pandas.read_csv unchanged.
    """
    new_file = not append or not os.path.exists(path)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "a" if append else "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if new_file:
            writer.writerow(header)
        writer.writerows(rows)


def _write_text(path: str, text: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
