"""Utility helpers for LIBERO evaluation script.

This module intentionally contains only side-effect-light helper functions so the
main evaluation script can stay focused on rollout logic.
"""

from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


def log_message(message: str, log_file=None):
    """Log a message to console and optionally to a log file."""
    logger.info(message)
    if log_file:
        log_file.write(message + "\n")
        log_file.flush()


def write_resume_episode_marker(
    task_id: int,
    episode_idx: int,
    ran_episode: bool,
    success: bool,
    infer_time_s: float,
    infer_count: int,
    reanchor_count: int,
    full_infer_time_s: float = 0.0,
    full_infer_count: int = 0,
    light_infer_time_s: float = 0.0,
    light_infer_count: int = 0,
    log_file=None,
):
    """Write a machine-readable progress marker line for robust resume."""
    marker = (
        "[RESUME_EPISODE] "
        f"task_id={task_id} "
        f"episode_idx={episode_idx} "
        f"ran={int(ran_episode)} "
        f"success={int(success)} "
        f"infer_time_s={float(infer_time_s):.9f} "
        f"infer_count={int(infer_count)} "
        f"reanchor_count={int(reanchor_count)} "
        f"full_infer_time_s={float(full_infer_time_s):.9f} "
        f"full_infer_count={int(full_infer_count)} "
        f"light_infer_time_s={float(light_infer_time_s):.9f} "
        f"light_infer_count={int(light_infer_count)}"
    )
    log_message(marker, log_file)


def _build_resume_summary_from_episode_records(task_episode_records: dict, num_tasks: int) -> dict:
    """Convert parsed per-task episode records into aggregate resume summary."""
    task_states = {}
    total_episodes = 0
    total_successes = 0
    total_infer_time_s = 0.0
    total_infer_count = 0
    total_reanchor_count = 0
    total_full_infer_time_s = 0.0
    total_full_infer_count = 0
    total_light_infer_time_s = 0.0
    total_light_infer_count = 0

    for task_id in range(num_tasks):
        episode_records = task_episode_records.get(task_id, {})
        processed_episode_indices = sorted(episode_records.keys())

        task_episodes = sum(1 for rec in episode_records.values() if rec["ran"])
        task_successes = sum(1 for rec in episode_records.values() if rec["ran"] and rec["success"])
        task_infer_time_s = sum(float(rec["infer_time_s"]) for rec in episode_records.values() if rec["ran"])
        task_infer_count = sum(int(rec["infer_count"]) for rec in episode_records.values() if rec["ran"])
        task_reanchor_count = sum(int(rec["reanchor_count"]) for rec in episode_records.values() if rec["ran"])
        task_full_infer_time_s = sum(float(rec.get("full_infer_time_s", 0.0)) for rec in episode_records.values() if rec["ran"])
        task_full_infer_count = sum(int(rec.get("full_infer_count", 0)) for rec in episode_records.values() if rec["ran"])
        task_light_infer_time_s = sum(float(rec.get("light_infer_time_s", 0.0)) for rec in episode_records.values() if rec["ran"])
        task_light_infer_count = sum(int(rec.get("light_infer_count", 0)) for rec in episode_records.values() if rec["ran"])

        task_states[task_id] = {
            "processed_episode_indices": processed_episode_indices,
            "task_episodes": int(task_episodes),
            "task_successes": int(task_successes),
            "task_infer_time_s": float(task_infer_time_s),
            "task_infer_count": int(task_infer_count),
            "task_reanchor_count": int(task_reanchor_count),
            "task_full_infer_time_s": float(task_full_infer_time_s),
            "task_full_infer_count": int(task_full_infer_count),
            "task_light_infer_time_s": float(task_light_infer_time_s),
            "task_light_infer_count": int(task_light_infer_count),
        }

        total_episodes += int(task_episodes)
        total_successes += int(task_successes)
        total_infer_time_s += float(task_infer_time_s)
        total_infer_count += int(task_infer_count)
        total_reanchor_count += int(task_reanchor_count)
        total_full_infer_time_s += float(task_full_infer_time_s)
        total_full_infer_count += int(task_full_infer_count)
        total_light_infer_time_s += float(task_light_infer_time_s)
        total_light_infer_count += int(task_light_infer_count)

    return {
        "task_states": task_states,
        "total_episodes": int(total_episodes),
        "total_successes": int(total_successes),
        "total_infer_time_s": float(total_infer_time_s),
        "total_infer_count": int(total_infer_count),
        "total_reanchor_count": int(total_reanchor_count),
        "total_full_infer_time_s": float(total_full_infer_time_s),
        "total_full_infer_count": int(total_full_infer_count),
        "total_light_infer_time_s": float(total_light_infer_time_s),
        "total_light_infer_count": int(total_light_infer_count),
    }


def _load_resume_state_from_legacy_log(log_path: str, num_tasks: int, task_suite, log_file=None) -> dict:
    """Best-effort parser for pre-marker logs."""
    task_description_to_id = {}
    for task_id in range(num_tasks):
        task = task_suite.get_task(task_id)
        task_description_to_id[str(task.language).strip()] = task_id

    segment_episode_records = []
    segment_max_total_episodes = []
    segment_episode_count = []

    current_segment_records = {}
    current_segment_max_total_episodes = 0
    current_task_id = None
    pending_episode = None

    def _start_new_segment():
        nonlocal current_segment_records, current_segment_max_total_episodes, current_task_id, pending_episode
        segment_episode_records.append(current_segment_records)
        segment_max_total_episodes.append(int(current_segment_max_total_episodes))
        segment_episode_count.append(
            sum(1 for task_map in current_segment_records.values() for rec in task_map.values() if rec["ran"])
        )
        current_segment_records = {}
        current_segment_max_total_episodes = 0
        current_task_id = None
        pending_episode = None

    def _upsert_record(task_id: int, episode_idx: int, record: dict):
        if task_id < 0 or task_id >= num_tasks:
            return
        current_segment_records.setdefault(task_id, {})[episode_idx] = record

    task_line_pattern = re.compile(r"^Task:\s*(?P<task_desc>.+?)\s*$")
    start_episode_pattern = re.compile(r"^Starting episode\s+(?P<episode_num>\d+)\.\.\.\s*$")
    success_pattern = re.compile(r"^Success:\s*(?P<success>True|False)\s*$")
    infer_pattern = re.compile(
        r"^Episode inference stats:\s*"
        r"time_s=(?P<infer_time_s>[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?),\s*"
        r"calls=(?P<infer_count>\d+)"
        r"(?:,\s*avg_time_s=[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)?"
        r"(?:,\s*reanchor=(?P<reanchor_count>\d+))?\s*$"
    )
    skip_pattern = re.compile(r"^Skipping task\s+(?P<task_id>\d+)\s+episode\s+(?P<episode_idx>\d+)\s+.*$")
    total_episode_pattern = re.compile(r"^# episodes completed so far:\s*(?P<count>\d+)\s*$")

    with open(log_path, "r") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue

            if line.startswith("Task suite:"):
                _start_new_segment()
                continue

            m = task_line_pattern.match(line)
            if m is not None:
                task_desc = m.group("task_desc").strip()
                current_task_id = task_description_to_id.get(task_desc)
                continue

            m = skip_pattern.match(line)
            if m is not None:
                task_id = int(m.group("task_id"))
                episode_idx = int(m.group("episode_idx"))
                _upsert_record(
                    task_id,
                    episode_idx,
                    {
                        "ran": False,
                        "success": False,
                        "infer_time_s": 0.0,
                        "infer_count": 0,
                        "reanchor_count": 0,
                        "full_infer_time_s": 0.0,
                        "full_infer_count": 0,
                        "light_infer_time_s": 0.0,
                        "light_infer_count": 0,
                    },
                )
                continue

            m = start_episode_pattern.match(line)
            if m is not None:
                pending_episode = {
                    "task_id": current_task_id,
                    "episode_idx": int(m.group("episode_num")) - 1,
                    "success": None,
                }
                continue

            m = success_pattern.match(line)
            if m is not None and pending_episode is not None:
                pending_episode["success"] = m.group("success") == "True"
                continue

            m = infer_pattern.match(line)
            if m is not None and pending_episode is not None:
                if pending_episode["task_id"] is not None and pending_episode["episode_idx"] >= 0:
                    _upsert_record(
                        int(pending_episode["task_id"]),
                        int(pending_episode["episode_idx"]),
                        {
                            "ran": True,
                            "success": bool(pending_episode.get("success", False)),
                            "infer_time_s": float(m.group("infer_time_s")),
                            "infer_count": int(m.group("infer_count")),
                            "reanchor_count": int(m.group("reanchor_count") or 0),
                            "full_infer_time_s": 0.0,
                            "full_infer_count": 0,
                            "light_infer_time_s": 0.0,
                            "light_infer_count": 0,
                        },
                    )
                pending_episode = None
                continue

            m = total_episode_pattern.match(line)
            if m is not None:
                current_segment_max_total_episodes = max(current_segment_max_total_episodes, int(m.group("count")))
                continue

    _start_new_segment()

    if len(segment_episode_records) == 0:
        return {
            "task_states": {task_id: {
                "processed_episode_indices": [],
                "task_episodes": 0,
                "task_successes": 0,
                "task_infer_time_s": 0.0,
                "task_infer_count": 0,
                "task_reanchor_count": 0,
                "task_full_infer_time_s": 0.0,
                "task_full_infer_count": 0,
                "task_light_infer_time_s": 0.0,
                "task_light_infer_count": 0,
            } for task_id in range(num_tasks)},
            "total_episodes": 0,
            "total_successes": 0,
            "total_infer_time_s": 0.0,
            "total_infer_count": 0,
            "total_reanchor_count": 0,
            "total_full_infer_time_s": 0.0,
            "total_full_infer_count": 0,
            "total_light_infer_time_s": 0.0,
            "total_light_infer_count": 0,
            "legacy_segments": 0,
            "legacy_best_segment_idx": -1,
        }

    best_segment_idx = max(
        range(len(segment_episode_records)),
        key=lambda i: (segment_max_total_episodes[i], segment_episode_count[i], i),
    )
    best_records = segment_episode_records[best_segment_idx]
    summary = _build_resume_summary_from_episode_records(best_records, num_tasks)
    summary.update(
        {
            "legacy_segments": len(segment_episode_records),
            "legacy_best_segment_idx": int(best_segment_idx),
        }
    )
    return summary


def load_resume_state_from_log(log_path: str, num_tasks: int, task_suite, log_file=None) -> dict:
    """Load resumable evaluation state from a prior log file."""
    resume_path = Path(log_path)
    assert resume_path.exists(), f"resume_from_log_path does not exist: {log_path}"

    marker_pattern = re.compile(
        r"\[RESUME_EPISODE\]\s+"
        r"task_id=(?P<task_id>\d+)\s+"
        r"episode_idx=(?P<episode_idx>\d+)\s+"
        r"ran=(?P<ran>[01])\s+"
        r"success=(?P<success>[01])\s+"
        r"infer_time_s=(?P<infer_time_s>[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s+"
        r"infer_count=(?P<infer_count>\d+)\s+"
        r"reanchor_count=(?P<reanchor_count>\d+)"
        r"(?:\s+full_infer_time_s=(?P<full_infer_time_s>[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?))?"
        r"(?:\s+full_infer_count=(?P<full_infer_count>\d+))?"
        r"(?:\s+light_infer_time_s=(?P<light_infer_time_s>[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?))?"
        r"(?:\s+light_infer_count=(?P<light_infer_count>\d+))?"
    )

    task_episode_records = {}
    with open(resume_path, "r") as f:
        for line in f:
            match = marker_pattern.search(line)
            if match is None:
                continue

            task_id = int(match.group("task_id"))
            episode_idx = int(match.group("episode_idx"))
            if task_id < 0 or task_id >= num_tasks:
                continue

            task_episode_records.setdefault(task_id, {})[episode_idx] = {
                "ran": bool(int(match.group("ran"))),
                "success": bool(int(match.group("success"))),
                "infer_time_s": float(match.group("infer_time_s")),
                "infer_count": int(match.group("infer_count")),
                "reanchor_count": int(match.group("reanchor_count")),
                "full_infer_time_s": float(match.group("full_infer_time_s") or 0.0),
                "full_infer_count": int(match.group("full_infer_count") or 0),
                "light_infer_time_s": float(match.group("light_infer_time_s") or 0.0),
                "light_infer_count": int(match.group("light_infer_count") or 0),
            }

    resumed_summary = _build_resume_summary_from_episode_records(task_episode_records, num_tasks)

    matched_marker_count = sum(len(v) for v in task_episode_records.values())
    if matched_marker_count == 0:
        legacy_summary = _load_resume_state_from_legacy_log(log_path, num_tasks, task_suite, log_file)
        if int(legacy_summary["total_episodes"]) > 0:
            log_message(
                f"Resume requested from {log_path} without [RESUME_EPISODE] markers. "
                f"Recovered legacy state from segment {legacy_summary['legacy_best_segment_idx'] + 1}/"
                f"{legacy_summary['legacy_segments']}: "
                f"episodes={legacy_summary['total_episodes']}, successes={legacy_summary['total_successes']}",
                log_file,
            )
            return {
                "task_states": legacy_summary["task_states"],
                "total_episodes": legacy_summary["total_episodes"],
                "total_successes": legacy_summary["total_successes"],
                "total_infer_time_s": legacy_summary["total_infer_time_s"],
                "total_infer_count": legacy_summary["total_infer_count"],
                "total_reanchor_count": legacy_summary["total_reanchor_count"],
                "total_full_infer_time_s": legacy_summary.get("total_full_infer_time_s", 0.0),
                "total_full_infer_count": legacy_summary.get("total_full_infer_count", 0),
                "total_light_infer_time_s": legacy_summary.get("total_light_infer_time_s", 0.0),
                "total_light_infer_count": legacy_summary.get("total_light_infer_count", 0),
            }

        log_message(
            f"Resume requested from {log_path}, but no [RESUME_EPISODE] markers were found and legacy recovery "
            f"found no completed episodes. Starting from scratch.",
            log_file,
        )
    else:
        log_message(
            f"Loaded resume state from {log_path}: markers={matched_marker_count}, "
            f"episodes={resumed_summary['total_episodes']}, successes={resumed_summary['total_successes']}",
            log_file,
        )

    return resumed_summary


def _parse_int_csv(value: str) -> set[int]:
    out = set()
    if not value:
        return out
    for token in value.split(","):
        token = token.strip()
        if not token:
            continue
        try:
            idx = int(token)
        except ValueError:
            continue
        if idx >= 0:
            out.add(idx)
    return out


def _safe_task_slug(task_description: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9_-]+", "_", task_description.strip().lower())
    return slug.strip("_") or "task"


def _save_context_frames_for_episode(
    cfg: Any,
    replay_images,
    task_id: int,
    episode_idx: int,
    task_description: str,
    focus_frames: set[int],
    log_file=None,
) -> None:
    if not cfg.save_context_frames or len(replay_images) == 0 or len(focus_frames) == 0:
        return

    window = max(0, int(cfg.context_frame_window))
    task_slug = _safe_task_slug(task_description)
    ep_dir = Path(cfg.context_frame_dir) / f"task_{task_id:03d}_{task_slug}" / f"episode_{episode_idx:03d}"
    ep_dir.mkdir(parents=True, exist_ok=True)

    metadata = {
        "task_id": int(task_id),
        "episode_idx": int(episode_idx),
        "task_description": task_description,
        "window": int(window),
        "focus_frames": sorted(int(x) for x in focus_frames),
        "saved_files": [],
    }

    total_frames = len(replay_images)
    for focus in sorted(focus_frames):
        start = max(0, focus - window)
        end = min(total_frames - 1, focus + window)
        for frame_idx in range(start, end + 1):
            img = replay_images[frame_idx]
            out_name = f"focus_{focus:04d}__frame_{frame_idx:04d}.png"
            out_path = ep_dir / out_name
            if not out_path.exists():
                Image.fromarray(np.asarray(img)).save(out_path)
            metadata["saved_files"].append(out_name)

    meta_path = ep_dir / "context_frames_metadata.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    log_message(
        f"Saved context frames for task_id={task_id}, episode_idx={episode_idx} at {ep_dir} "
        f"(focus={sorted(focus_frames)}, window={window})",
        log_file,
    )


def load_initial_states(cfg: Any, task_suite, task_id: int, log_file=None):
    """Load initial states for the given task."""
    initial_states = task_suite.get_task_init_states(task_id)

    if cfg.initial_states_path != "DEFAULT":
        with open(cfg.initial_states_path, "r") as f:
            all_initial_states = json.load(f)
        log_message(f"Using initial states from {cfg.initial_states_path}", log_file)
        return initial_states, all_initial_states
    else:
        log_message("Using default initial states", log_file)
        return initial_states, None
