"""Durable, single-process runner for the frozen Wall decision-audit manifest.

The runner owns only runtime state under ``run_dir``.  ``protocol.json`` and
``manifest.json`` are read-only inputs supplied by the study owner.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import math
import os
import platform
import shutil
import socket
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


SCHEMA_VERSION = 1
MODELS = ("jepa", "dino")


class RunnerError(RuntimeError):
    pass


class LockError(RunnerError):
    pass


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise RunnerError(f"invalid JSON: {path}: {exc}") from exc


def atomic_write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def atomic_write_json(path: Path, value: Any) -> None:
    atomic_write_bytes(path, canonical(value) + b"\n")


def append_event(path: Path, event: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(event, sort_keys=True, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def safe_name(value: str) -> str:
    text = "".join(char if char.isalnum() or char in "._-" else "_" for char in value)
    text = text.strip("._") or "item"
    return text[:100]


def item_key(item: dict[str, Any]) -> str:
    return f"{item['case_id']}::{item['case_seed']}::{item['menu_seed']}::{item['model']}"


def _pid_alive(pid: int) -> bool:
    if pid == os.getpid():
        return True
    if os.name != "nt":
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return False
        except PermissionError:
            return True
        return True
    # Windows has no harmless signal-0 equivalent. Query the process state.
    import ctypes

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    handle = kernel32.OpenProcess(0x1000, False, pid)  # PROCESS_QUERY_LIMITED_INFORMATION
    if not handle:
        error = ctypes.get_last_error()
        if error in (87, 116, 128):  # invalid parameter / no such process
            return False
        return True  # access denied is unverifiable; refuse stale removal
    try:
        exit_code = ctypes.c_ulong()
        if not kernel32.GetExitCodeProcess(handle, ctypes.byref(exit_code)):
            return True
        return exit_code.value == 259  # STILL_ACTIVE
    finally:
        kernel32.CloseHandle(handle)


class RunLock:
    def __init__(self, run_dir: Path):
        self.path = run_dir / ".runner.lock"
        self.payload = {"pid": os.getpid(), "hostname": socket.gethostname(), "started_at": utc_now()}
        self.held = False

    def acquire(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        try:
            fd = os.open(str(self.path), os.O_WRONLY | os.O_CREAT | os.O_EXCL)
        except FileExistsError as exc:
            old: Any = None
            try:
                old = read_json(self.path)
            except RunnerError:
                pass
            detail = f" existing lock={old!r}" if old is not None else " (lock contents unreadable)"
            raise LockError(f"run directory is already locked{detail}; inspect and use --unlock-stale only after verifying the PID is dead") from exc
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(self.payload, handle, sort_keys=True)
            handle.flush()
            os.fsync(handle.fileno())
        self.held = True

    def release(self) -> None:
        if self.held:
            try:
                self.path.unlink()
            except FileNotFoundError:
                pass
            self.held = False

    def __enter__(self) -> "RunLock":
        self.acquire()
        return self

    def __exit__(self, *_: Any) -> None:
        self.release()


def unlock_stale(run_dir: Path) -> str:
    path = run_dir / ".runner.lock"
    if not path.exists():
        return "no lock present"
    payload = read_json(path)
    pid = payload.get("pid") if isinstance(payload, dict) else None
    hostname = payload.get("hostname") if isinstance(payload, dict) else None
    if hostname != socket.gethostname():
        raise LockError(f"refusing to remove lock from another or unknown host (hostname={hostname!r})")
    if not isinstance(pid, int) or _pid_alive(pid):
        raise LockError(f"refusing to remove active or unverifiable lock (pid={pid!r})")
    path.unlink()
    return f"removed stale lock for pid {pid}"


def load_items(manifest_path: Path) -> list[dict[str, Any]]:
    raw = read_json(manifest_path)
    items = raw if isinstance(raw, list) else (raw.get("items") or raw.get("rows")) if isinstance(raw, dict) else None
    if not isinstance(items, list) or not items:
        raise RunnerError("manifest must contain a non-empty 'rows' or 'items' array")
    validated: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, item in enumerate(items):
        if not isinstance(item, dict) or set(("case_id", "case_seed", "menu_seed", "model")) - item.keys():
            raise RunnerError(f"manifest item {index} lacks case_id/case_seed/menu_seed/model")
        if item["model"] not in MODELS:
            raise RunnerError(f"manifest item {index} has unsupported model {item['model']!r}")
        if isinstance(item["case_seed"], bool) or not isinstance(item["case_seed"], int):
            raise RunnerError(f"manifest item {index} case_seed must be an integer")
        if isinstance(item["menu_seed"], bool) or not isinstance(item["menu_seed"], int):
            raise RunnerError(f"manifest item {index} menu_seed must be an integer")
        key = item_key(item)
        if key in seen:
            raise RunnerError(f"duplicate manifest item {key}")
        seen.add(key)
        validated.append(dict(item))
    return validated


def checkpoint_map(args: argparse.Namespace, manifest_models: set[str] | None = None) -> dict[str, Path]:
    paths = {"jepa": args.jepa_checkpoint, "dino": args.dino_checkpoint}
    selected = set(args.models or manifest_models or MODELS)
    result: dict[str, Path] = {}
    for model in selected:
        path = paths[model]
        if path is None:
            raise RunnerError(f"--{model}-checkpoint is required for selected model {model}")
        path = path.resolve()
        if not path.is_file():
            raise RunnerError(f"checkpoint is not a file: {path}")
        result[model] = path
    return result


def gpu_provenance() -> dict[str, Any]:
    value: dict[str, Any] = {
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "device": os.environ.get("WALL_AUDIT_DEVICE", "unspecified"),
    }
    try:
        proc = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10, check=False,
        )
        value["nvidia_smi"] = proc.stdout.strip().splitlines()
    except (OSError, subprocess.SubprocessError):
        value["nvidia_smi"] = []
    return value


def compatibility(run_dir: Path, args: argparse.Namespace, checkpoints: dict[str, Path]) -> dict[str, Any]:
    protocol = run_dir / "protocol.json"
    manifest = run_dir / "manifest.json"
    if not protocol.is_file() or not manifest.is_file():
        raise RunnerError(f"run directory must already contain protocol.json and manifest.json: {run_dir}")
    protocol_hash = sha256_file(protocol)
    manifest_hash = sha256_file(manifest)
    runner_hash = sha256_file(Path(__file__).resolve())
    code_files = {"runner": runner_hash}
    child = Path(__file__).with_name("wall_decision_audit.py")
    if child.is_file():
        code_files["child"] = sha256_file(child)
    code_hash = sha256_bytes(canonical(code_files))
    # Paths are local to each machine.  Bytes, names, and sizes are portable.
    checkpoint_hashes = {model: {"name": path.name, "size": path.stat().st_size, "sha256": sha256_file(path)} for model, path in sorted(checkpoints.items())}
    body = {"protocol_sha256": protocol_hash, "manifest_sha256": manifest_hash, "runner_sha256": runner_hash, "code_sha256": code_hash, "checkpoints": checkpoint_hashes}
    body["compatibility_sha256"] = sha256_bytes(canonical(body))
    return body


def validate_result(item: dict[str, Any], value: Any) -> None:
    if not isinstance(value, dict):
        raise RunnerError("child evidence must be a JSON object")
    for field in ("case_id", "case_seed", "menu_seed", "model"):
        if field not in value or value[field] != item[field]:
            raise RunnerError(f"child evidence identity mismatch for {field}: expected {item[field]!r}")
    if "decision_id" in item and value.get("decision_id") != item["decision_id"]:
        raise RunnerError(f"child evidence identity mismatch for decision_id: expected {item['decision_id']!r}")
    if value.get("status") != "PASS" or value.get("scientific_result") is not True or value.get("candidates") != 300:
        raise RunnerError("child evidence is not a successful 300-candidate scientific result")
    details = value.get("candidates_detail")
    if not isinstance(details, list) or len(details) != 300:
        raise RunnerError("child evidence candidates_detail must contain exactly 300 rows")
    for index, row in enumerate(details):
        if not isinstance(row, dict) or row.get("candidate") != index:
            raise RunnerError(f"candidate detail index mismatch at {index}")
        for field in ("predicted_cost_P", "encoded_real_cost_O", "native_terminal_distance"):
            if not isinstance(row.get(field), (int, float)) or not math.isfinite(float(row[field])):
                raise RunnerError(f"candidate detail {field} must be finite at {index}")
        if not isinstance(row.get("native_terminal_success"), bool):
            raise RunnerError(f"candidate detail native_terminal_success must be boolean at {index}")


def verify_sidecar(value: dict[str, Any], search_dir: Path | None = None) -> Path | None:
    """Verify an evaluator sidecar, tolerating a changed absolute root after transfer."""
    origin = value.get("arrays_sidecar")
    expected = value.get("arrays_sidecar_sha256")
    if origin is None and expected is None:
        return None
    if not isinstance(origin, str) or not isinstance(expected, str) or len(expected) != 64:
        raise RunnerError("arrays_sidecar and arrays_sidecar_sha256 must be recorded together")
    candidates = [Path(origin)]
    if search_dir is not None:
        candidates.append(search_dir / Path(origin).name)
        candidates.extend(search_dir.rglob(Path(origin).name))
    seen: set[Path] = set()
    for candidate in candidates:
        candidate = candidate.resolve()
        if candidate in seen or not candidate.is_file():
            continue
        seen.add(candidate)
        if sha256_file(candidate) == expected:
            return candidate
    raise RunnerError(f"arrays sidecar is missing or hash-mismatched after relocation: {origin}")


def result_path(run_dir: Path, item: dict[str, Any]) -> Path:
    label = item.get("decision_id") or f"{item['case_id']}__seed{item['case_seed']}__menu{item['menu_seed']}__{item['model']}"
    return run_dir / "results" / f"{safe_name(str(label))}.json"


def item_dir(run_dir: Path, index: int, item: dict[str, Any]) -> Path:
    return run_dir / "attempts" / f"{index:05d}_{safe_name(item_key(item))}"


def latest_attempt(directory: Path) -> dict[str, Any] | None:
    values = []
    for path in directory.glob("*/attempt.json"):
        try:
            value = read_json(path)
        except RunnerError:
            continue
        if isinstance(value, dict):
            values.append(value)
    return max(values, key=lambda value: int(value.get("attempt", 0)), default=None)


def write_final_evidence(path: Path, raw: bytes, value: Any) -> None:
    if path.exists():
        existing = read_json(path)
        for field in ("case_id", "case_seed", "menu_seed", "model"):
            if not isinstance(existing, dict) or existing.get(field) != value.get(field):
                raise RunnerError(f"existing evidence identity mismatch for {field}")
        if "decision_id" in value and (not isinstance(existing, dict) or existing.get("decision_id") != value["decision_id"]):
            raise RunnerError("existing evidence identity mismatch for decision_id")
        if sha256_bytes(canonical(existing)) != sha256_bytes(canonical(value)):
            raise RunnerError(f"conflicting overlap result already exists: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)  # atomic create-without-overwrite
        except FileExistsError:
            return write_final_evidence(path, raw, value)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def status_payload(*, started: float, selected: int, complete: int, failed: int, current: str | None, durations: list[float], provenance: dict[str, Any], compatibility_sha256: str | None = None, state: str | None = None) -> dict[str, Any]:
    average = sum(durations) / len(durations) if durations else None
    remaining = max(0, selected - complete - failed)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "updated_at": utc_now(),
        "hostname": socket.gethostname(),
        "status": state or ("complete_with_failures" if failed and complete + failed == selected else "complete" if complete + failed == selected else "running"),
        "total": selected,
        "complete": complete,
        "failed": failed,
        "pending": remaining,
        "current_case": current,
        "elapsed_seconds": round(max(0.0, time.monotonic() - started), 3),
        "eta_seconds": round(remaining * average, 3) if average is not None else None,
        "provenance": provenance,
    }

    if compatibility_sha256:
        payload["compatibility_sha256"] = compatibility_sha256
    return payload


@contextlib.contextmanager
def keep_awake():
    """Prevent display-independent system sleep during an unattended Windows run."""
    if os.name != "nt":
        yield
        return
    import ctypes
    continuous = 0x80000000
    required = 0x00000001  # ES_SYSTEM_REQUIRED; do not alter display settings
    try:
        ctypes.windll.kernel32.SetThreadExecutionState(continuous | required)
    except (AttributeError, OSError):
        yield
        return
    try:
        yield
    finally:
        try:
            ctypes.windll.kernel32.SetThreadExecutionState(continuous)
        except (AttributeError, OSError):
            pass


def run_one(*, run_dir: Path, item_index: int, item: dict[str, Any], checkpoint: Path, args: argparse.Namespace, event_path: Path) -> tuple[str, float]:
    destination = item_dir(run_dir, item_index, item)
    destination.mkdir(parents=True, exist_ok=True)
    attempts = [int(path.parent.name) for path in destination.glob("*/attempt.json") if path.parent.name.isdigit()]
    attempt_number = max(attempts, default=0) + 1
    attempt_dir = destination / f"{attempt_number:04d}"
    attempt_dir.mkdir(parents=True)
    output = attempt_dir / "child_output.json"
    stdout_path = attempt_dir / "stdout.log"
    stderr_path = attempt_dir / "stderr.log"
    command = [
        sys.executable, "-m", "experiments.wall_decision_audit",
        "--official-root", str(args.official_root.resolve()), "--model", item["model"],
        "--checkpoint", str(checkpoint), "--case-seed", str(item["case_seed"]),
        "--menu-seed", str(item["menu_seed"]), "--output", str(output),
        "--device", args.device, "--candidates", "300", "--case-id", str(item["case_id"]),
    ]
    if "decision_id" in item:
        command += ["--decision-id", str(item["decision_id"])]
    started = time.monotonic()
    metadata: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION, "item_index": item_index, "item": item,
        "attempt": attempt_number, "status": "running", "started_at": utc_now(),
        "command": command, "stdout": str(stdout_path), "stderr": str(stderr_path), "output": str(output),
        "retry_reason": args.retry_reason if args.retry_failed or args.retry_infrastructure else None,
    }
    atomic_write_json(attempt_dir / "attempt.json", metadata)
    append_event(event_path, {"at": utc_now(), "event": "case_started", "item": item, "attempt": attempt_number})
    with stdout_path.open("w", encoding="utf-8", newline="\n") as stdout, stderr_path.open("w", encoding="utf-8", newline="\n") as stderr:
        process = None
        try:
            process = subprocess.Popen(command, stdout=stdout, stderr=stderr, cwd=str(Path.cwd()))
            metadata.update({"pid": getattr(process, "pid", None), "pid_recorded_at": utc_now()})
            atomic_write_json(attempt_dir / "attempt.json", metadata)
            code = process.wait()
        except OSError as exc:
            duration = time.monotonic() - started
            metadata.update({"status": "failed", "failure_class": "infrastructure", "error": str(exc), "finished_at": utc_now(), "duration_seconds": duration})
            atomic_write_json(attempt_dir / "attempt.json", metadata)
            append_event(event_path, {"at": utc_now(), "event": "case_failed", "item": item, "attempt": attempt_number, "failure_class": "infrastructure", "error": str(exc)})
            return "failed", duration
        except KeyboardInterrupt:
            try:
                if process is not None:
                    process.terminate()
                    process.wait(timeout=10)
            except (OSError, subprocess.SubprocessError):
                pass
            metadata.update({"status": "interrupted", "finished_at": utc_now(), "returncode": None})
            atomic_write_json(attempt_dir / "attempt.json", metadata)
            append_event(event_path, {"at": utc_now(), "event": "case_interrupted", "item": item, "attempt": attempt_number})
            raise
    duration = time.monotonic() - started
    if code != 0:
        metadata.update({"status": "failed", "failure_class": "infrastructure" if code in (75, 125) else "scientific_or_child", "returncode": code, "finished_at": utc_now(), "duration_seconds": duration})
        atomic_write_json(attempt_dir / "attempt.json", metadata)
        append_event(event_path, {"at": utc_now(), "event": "case_failed", "item": item, "attempt": attempt_number, "returncode": code, "failure_class": metadata["failure_class"]})
        return "failed", duration
    try:
        raw = output.read_bytes()
        value = json.loads(raw.decode("utf-8"))
        validate_result(item, value)
        verify_sidecar(value, attempt_dir)
        write_final_evidence(result_path(run_dir, item), raw, value)
    except (OSError, ValueError, RunnerError) as exc:
        metadata.update({"status": "failed", "failure_class": "scientific_or_child", "error": str(exc), "finished_at": utc_now(), "duration_seconds": duration})
        atomic_write_json(attempt_dir / "attempt.json", metadata)
        append_event(event_path, {"at": utc_now(), "event": "case_failed", "item": item, "attempt": attempt_number, "failure_class": metadata["failure_class"], "error": str(exc)})
        return "failed", duration
    metadata.update({"status": "complete", "finished_at": utc_now(), "duration_seconds": duration, "result": str(result_path(run_dir, item)), "result_sha256": sha256_bytes(canonical(value))})
    atomic_write_json(attempt_dir / "attempt.json", metadata)
    append_event(event_path, {"at": utc_now(), "event": "case_complete", "item": item, "attempt": attempt_number, "result": str(result_path(run_dir, item))})
    return "complete", duration


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--run-dir", type=Path, required=True)
    result.add_argument("--official-root", type=Path, required=True)
    result.add_argument("--jepa-checkpoint", type=Path)
    result.add_argument("--dino-checkpoint", type=Path)
    result.add_argument("--device", default="cuda")
    result.add_argument("--shard-index", type=int, default=0)
    result.add_argument("--shard-count", type=int, default=1)
    result.add_argument("--models", nargs="+", choices=MODELS)
    result.add_argument("--retry-failed", action="store_true")
    result.add_argument("--retry-infrastructure", action="store_true")
    result.add_argument("--retry-reason")
    result.add_argument("--unlock-stale", action="store_true")
    result.add_argument("--pause-file", type=Path, help="pause cleanly before the next case; defaults to RUN_DIR/PAUSE")
    result.add_argument("--min-free-gib", type=float, default=25.0, help="stop before a new case when run-drive free space is below this amount")
    return result


def main(argv: Iterable[str] | None = None) -> int:
    args = parser().parse_args(list(argv) if argv is not None else None)
    run_dir = args.run_dir.resolve()
    if args.unlock_stale:
        print(unlock_stale(run_dir))
        return 0
    if args.shard_count < 1 or not 0 <= args.shard_index < args.shard_count:
        raise RunnerError("shard index must be in [0, shard_count)")
    if (args.retry_failed or args.retry_infrastructure) and not args.retry_reason:
        raise RunnerError("--retry-failed/--retry-infrastructure requires --retry-reason")
    items = load_items(run_dir / "manifest.json")
    checkpoints = checkpoint_map(args, {str(item["model"]) for item in items})
    selected_models = set(args.models or MODELS)
    case_ids = sorted({str(item["case_id"]) for item in items})
    case_numbers = {case_id: index for index, case_id in enumerate(case_ids)}
    selected = [(index, item) for index, item in enumerate(items) if item["model"] in selected_models and case_numbers[str(item["case_id"])] % args.shard_count == args.shard_index]
    run_dir.mkdir(parents=True, exist_ok=True)
    event_path = run_dir / "events.jsonl"
    with keep_awake(), RunLock(run_dir):
        compat = compatibility(run_dir, args, checkpoints)
        compatibility_path = run_dir / ".compatibility.json"
        if compatibility_path.exists() and read_json(compatibility_path) != compat:
            raise RunnerError("run compatibility mismatch: protocol, manifest, runner code, or checkpoint changed")
        atomic_write_json(compatibility_path, compat)
        provenance = {"hostname": socket.gethostname(), "platform": platform.platform(), "python": platform.python_version(), "executable": sys.executable, "gpu": gpu_provenance()}
        started = time.monotonic()
        durations: list[float] = []
        complete = failed = 0
        atomic_write_json(run_dir / "status.json", status_payload(started=started, selected=len(selected), complete=0, failed=0, current=None, durations=durations, provenance=provenance, compatibility_sha256=compat["compatibility_sha256"]))
        append_event(event_path, {"at": utc_now(), "event": "runner_started", "shard_index": args.shard_index, "shard_count": args.shard_count, "selected": len(selected), "compatibility_sha256": compat["compatibility_sha256"]})
        pause_file = (args.pause_file or (run_dir / "PAUSE")).resolve()
        for item_index, item in selected:
            if pause_file.exists():
                atomic_write_json(run_dir / "status.json", status_payload(started=started, selected=len(selected), complete=complete, failed=failed, current=None, durations=durations, provenance=provenance, compatibility_sha256=compat["compatibility_sha256"], state="paused"))
                append_event(event_path, {"at": utc_now(), "event": "runner_paused", "pause_file": str(pause_file), "next_item": item})
                return 0
            destination = result_path(run_dir, item)
            previous = latest_attempt(item_dir(run_dir, item_index, item))
            if destination.exists():
                value = read_json(destination)
                validate_result(item, value)
                verify_sidecar(value, item_dir(run_dir, item_index, item))
                evidence_hash = sha256_bytes(canonical(value))
                if previous and previous.get("result_sha256") and previous["result_sha256"] != evidence_hash:
                    raise RunnerError(f"final evidence hash conflicts with attempt receipt: {destination}")
                if previous and previous.get("status") == "running":
                    previous.update({"status": "complete", "finished_at": utc_now(), "recovered": True, "result": str(destination), "result_sha256": evidence_hash})
                    atomic_write_json(item_dir(run_dir, item_index, item) / f"{int(previous['attempt']):04d}" / "attempt.json", previous)
                    append_event(event_path, {"at": utc_now(), "event": "case_recovered", "item": item, "attempt": previous["attempt"]})
                complete += 1
                continue
            if previous and previous.get("status") == "running":
                old_pid = previous.get("pid")
                if isinstance(old_pid, int) and _pid_alive(old_pid):
                    raise RunnerError(f"previous child PID {old_pid} is still alive for {item_key(item)}; stop it before resuming")
                previous["status"] = "interrupted"
                previous["finished_at"] = utc_now()
                atomic_write_json(item_dir(run_dir, item_index, item) / f"{int(previous['attempt']):04d}" / "attempt.json", previous)
                append_event(event_path, {"at": utc_now(), "event": "stale_attempt_interrupted", "item": item, "attempt": previous["attempt"]})
            if previous and previous.get("status") == "failed":
                retry = args.retry_failed or (args.retry_infrastructure and previous.get("failure_class") == "infrastructure")
                if not retry:
                    failed += 1
                    continue
            free_gib = shutil.disk_usage(run_dir).free / (1024 ** 3)
            if free_gib < args.min_free_gib:
                atomic_write_json(run_dir / "status.json", status_payload(started=started, selected=len(selected), complete=complete, failed=failed, current=item_key(item), durations=durations, provenance=provenance, compatibility_sha256=compat["compatibility_sha256"], state="blocked_low_disk"))
                append_event(event_path, {"at": utc_now(), "event": "runner_blocked_low_disk", "item": item, "free_gib": free_gib, "min_free_gib": args.min_free_gib})
                return 0
            atomic_write_json(run_dir / "status.json", status_payload(started=started, selected=len(selected), complete=complete, failed=failed, current=item_key(item), durations=durations, provenance=provenance, compatibility_sha256=compat["compatibility_sha256"]))
            outcome, duration = run_one(run_dir=run_dir, item_index=item_index, item=item, checkpoint=checkpoints[item["model"]], args=args, event_path=event_path)
            durations.append(duration)
            if outcome == "complete":
                complete += 1
            else:
                failed += 1
            atomic_write_json(run_dir / "status.json", status_payload(started=started, selected=len(selected), complete=complete, failed=failed, current=None, durations=durations, provenance=provenance, compatibility_sha256=compat["compatibility_sha256"]))
        atomic_write_json(run_dir / "status.json", status_payload(started=started, selected=len(selected), complete=complete, failed=failed, current=None, durations=durations, provenance=provenance, compatibility_sha256=compat["compatibility_sha256"]))
        append_event(event_path, {"at": utc_now(), "event": "runner_finished", "complete": complete, "failed": failed, "selected": len(selected)})
    return 0 if failed == 0 else 2


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except RunnerError as exc:
        print(f"run_wall_audit: {exc}", file=sys.stderr)
        raise SystemExit(2)
