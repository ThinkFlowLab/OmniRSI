"""Local external-command runner for a prepared accelerator machine."""

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time
import uuid

from .contracts import RunSpec
from .evaluation import evaluate, read_metric
from .reporting import write_report


def _write_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def repository_identity(repo: str | Path, exclude: Path | None = None) -> dict:
    """Fingerprint Git revision, tracked diff and nonignored untracked files."""
    repo = Path(repo).resolve()
    identity = {"path": str(repo), "available": False}
    def git(*args: str) -> bytes:
        result = subprocess.run(["git", "-C", str(repo), *args], stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE, timeout=15, check=False)
        if result.returncode:
            raise ValueError(result.stderr.decode("utf-8", errors="replace").strip())
        return result.stdout
    try:
        git_root = Path(os.fsdecode(git("rev-parse", "--show-toplevel")).strip()).resolve()
        if git_root != repo:
            raise ValueError("repo must be the root of its Git checkout")
        paths = ["."]
        if exclude and Path(exclude).resolve().is_relative_to(git_root):
            relative = Path(exclude).resolve().relative_to(git_root).as_posix()
            paths.append(f":(top,exclude,literal){relative}")
        head = git("rev-parse", "HEAD").decode("ascii").strip()
        diff = git("diff", "--binary", "HEAD", "--", *paths)
        untracked = []
        for raw in git("ls-files", "--others", "--exclude-standard", "-z").split(b"\0"):
            if not raw:
                continue
            name = os.fsdecode(raw)
            path = repo / name
            if exclude and path.absolute().is_relative_to(Path(exclude).resolve()):
                continue
            if path.is_symlink():
                digest = hashlib.sha256(os.fsencode(os.readlink(path))).hexdigest()
            else:
                hasher = hashlib.sha256()
                with path.open("rb") as file:
                    for block in iter(lambda: file.read(1024 * 1024), b""):
                        hasher.update(block)
                digest = hasher.hexdigest()
            untracked.append({"path": name, "sha256": digest})
        dirty_data = {"diff_sha256": hashlib.sha256(diff).hexdigest(), "untracked": untracked}
        dirty_hash = hashlib.sha256(json.dumps(dirty_data, sort_keys=True).encode()).hexdigest()
        identity.update(available=True, head=head, git_root=str(git_root), dirty=bool(diff or untracked),
                        dirty_hash=dirty_hash, untracked=untracked,
                        snapshot_hash=hashlib.sha256(f"{head}:{dirty_hash}".encode()).hexdigest())
    except (OSError, ValueError, subprocess.TimeoutExpired) as exc:
        identity["error"] = str(exc)
    return identity


class _WindowsJob:
    """A native Job Object owns descendants; taskkill may lack sandbox privileges."""

    def __init__(self):
        import ctypes
        from ctypes import wintypes

        class BasicLimits(ctypes.Structure):
            _fields_ = [("PerProcessUserTimeLimit", ctypes.c_int64), ("PerJobUserTimeLimit", ctypes.c_int64),
                        ("LimitFlags", wintypes.DWORD), ("MinimumWorkingSetSize", ctypes.c_size_t),
                        ("MaximumWorkingSetSize", ctypes.c_size_t), ("ActiveProcessLimit", wintypes.DWORD),
                        ("Affinity", ctypes.c_size_t), ("PriorityClass", wintypes.DWORD),
                        ("SchedulingClass", wintypes.DWORD)]

        class IOCounters(ctypes.Structure):
            _fields_ = [(name, ctypes.c_uint64) for name in
                        ("ReadOperationCount", "WriteOperationCount", "OtherOperationCount",
                         "ReadTransferCount", "WriteTransferCount", "OtherTransferCount")]

        class ExtendedLimits(ctypes.Structure):
            _fields_ = [("BasicLimitInformation", BasicLimits), ("IoInfo", IOCounters),
                        ("ProcessMemoryLimit", ctypes.c_size_t), ("JobMemoryLimit", ctypes.c_size_t),
                        ("PeakProcessMemoryUsed", ctypes.c_size_t), ("PeakJobMemoryUsed", ctypes.c_size_t)]

        self.ctypes = ctypes
        self.kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        self.kernel.CreateJobObjectW.argtypes = [ctypes.c_void_p, wintypes.LPCWSTR]
        self.kernel.CreateJobObjectW.restype = wintypes.HANDLE
        self.kernel.SetInformationJobObject.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD]
        self.kernel.SetInformationJobObject.restype = wintypes.BOOL
        self.kernel.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
        self.kernel.AssignProcessToJobObject.restype = wintypes.BOOL
        self.kernel.CloseHandle.argtypes = [wintypes.HANDLE]
        self.kernel.CloseHandle.restype = wintypes.BOOL
        self.handle = self.kernel.CreateJobObjectW(None, None)
        if not self.handle:
            raise ctypes.WinError(ctypes.get_last_error())
        limits = ExtendedLimits()
        limits.BasicLimitInformation.LimitFlags = 0x2000  # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        if not self.kernel.SetInformationJobObject(self.handle, 9, ctypes.byref(limits), ctypes.sizeof(limits)):
            failure = ctypes.WinError(ctypes.get_last_error())
            self.close()
            raise failure

    def attach(self, process: subprocess.Popen) -> None:
        process_handle = getattr(process, "_handle", None)
        if process_handle is None:
            raise OSError("Windows subprocess handle unavailable for Job Object supervision")
        if not self.kernel.AssignProcessToJobObject(self.handle, int(process_handle)):
            raise self.ctypes.WinError(self.ctypes.get_last_error())

    def close(self) -> None:
        if self.handle:
            self.kernel.CloseHandle(self.handle)
            self.handle = None


def _stop_tree(process: subprocess.Popen, windows_job: _WindowsJob | None = None) -> None:
    if windows_job is not None:
        windows_job.close()
    elif os.name != "nt":
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    if process.poll() is None:
        process.kill()
    process.wait(timeout=10)


def _execute(command: tuple[str, ...], cwd: Path, trial_dir: Path,
             deadline: float, metric_key: str | None) -> dict:
    trial_dir.mkdir(parents=True)
    result_path = trial_dir / "result.json"
    argv = [arg.replace("{result}", str(result_path)) for arg in command]
    record = {"status": "pending", "argv": argv, "cwd": str(cwd), "returncode": None,
              "result_path": str(result_path), "stdout_path": str(trial_dir / "stdout.log"),
              "stderr_path": str(trial_dir / "stderr.log"), "metric": None, "metric_error": None}
    started, process, windows_job = time.monotonic(), None, None
    try:
        remaining = deadline - started
        if remaining <= 0:
            record.update(status="timeout", error="whole-run deadline exhausted before command start")
        else:
            environment = os.environ.copy()
            environment["OMNIRSI_RESULT_PATH"] = str(result_path)
            options = {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP} if os.name == "nt" else {"start_new_session": True}
            if os.name == "nt":
                windows_job = _WindowsJob()
            with (trial_dir / "stdout.log").open("wb") as stdout, (trial_dir / "stderr.log").open("wb") as stderr:
                process = subprocess.Popen(argv, cwd=cwd, env=environment, stdout=stdout, stderr=stderr,
                                           shell=False, **options)
                record["pid"] = process.pid
                record["process_group_id"] = process.pid if os.name != "nt" else None
                if windows_job is not None:
                    windows_job.attach(process)
                try:
                    process.wait(timeout=max(0.001, deadline - time.monotonic()))
                    record.update(status="completed" if process.returncode == 0 else "failed",
                                  returncode=process.returncode)
                except subprocess.TimeoutExpired:
                    _stop_tree(process, windows_job)
                    record.update(status="timeout", returncode=process.returncode, error="whole-run deadline exceeded")
                except KeyboardInterrupt:
                    _stop_tree(process, windows_job)
                    record.update(status="cancelled", returncode=process.returncode, error="interrupted by user")
    except (OSError, subprocess.SubprocessError) as exc:
        if process is not None and process.poll() is None:
            _stop_tree(process, windows_job)
        record.update(status="error", error=str(exc))
    finally:
        if windows_job is not None:
            windows_job.close()
        if process is not None and os.name != "nt":
            _stop_tree(process)
        record["elapsed_seconds"] = time.monotonic() - started
    if metric_key is not None:
        try:
            record["metric"] = read_metric(json.loads(result_path.read_text(encoding="utf-8")), metric_key)
        except (OSError, ValueError, TypeError, OverflowError) as exc:
            record["metric_error"] = str(exc)
    _write_json(trial_dir / "execution.json", record)
    return record


def run_experiment(spec: RunSpec, environment: dict) -> Path:
    """Run paired commands, preserve evidence and return a unique run directory."""
    spec.validate()
    started = time.monotonic()
    deadline = started + spec.max_minutes * 60
    run_dir = Path(spec.output_dir).resolve() / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:10])
    run_dir.mkdir(parents=True, exist_ok=False)
    output_root = run_dir.parent
    _write_json(run_dir / "run.lock.json", spec.to_dict())
    _write_json(run_dir / "environment.json", environment)
    baseline, candidate, validation = [], [], None
    before, after, source_checks = {}, {}, []
    state, error = "completed", None

    def event(phase: str, **details: object) -> None:
        record = {"time": datetime.now(timezone.utc).isoformat(), "phase": phase, **details}
        with (run_dir / "events.jsonl").open("a", encoding="utf-8") as file:
            file.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
        _write_json(run_dir / "status.json", {"state": "running", "phase": phase, "run_dir": str(run_dir)})

    repos = {"baseline": Path(spec.repo).resolve(), "candidate": Path(spec.candidate_repo or spec.repo).resolve()}

    def execute_checked(command: tuple[str, ...], label: str, trial_dir: Path, metric_key: str | None) -> dict:
        source_before = repository_identity(repos[label], output_root)
        record = _execute(command, repos[label], trial_dir, deadline, metric_key)
        source_after = repository_identity(repos[label], output_root)
        check = {"path": str(trial_dir), "side": label, "before": source_before, "after": source_after}
        source_checks.append(check)
        _write_json(trial_dir / "source.json", check)
        return record

    try:
        event("source_snapshot")
        before = {label: repository_identity(repo, output_root) for label, repo in repos.items()}
        _write_json(run_dir / "repositories.lock.json", before)
        commands = {"baseline": spec.baseline_command, "candidate": spec.candidate_command}
        for index in range(spec.repetitions):
            for label in (("baseline", "candidate") if index % 2 == 0 else ("candidate", "baseline")):
                event("measurement", side=label, trial=index + 1)
                record = execute_checked(commands[label], label, run_dir / label / f"trial-{index + 1:03d}", spec.metric_key)
                (baseline if label == "baseline" else candidate).append(record)
                if record["status"] == "cancelled":
                    state = "cancelled"
                    raise KeyboardInterrupt
                if record["status"] != "completed":
                    state = "timeout" if record["status"] == "timeout" else "failed"
                    break
            if state != "completed":
                break
        if spec.validation_command is not None and state == "completed":
            event("quality_validation")
            validation = execute_checked(spec.validation_command, "candidate", run_dir / "validation", None)
            if validation["status"] != "completed":
                state = "cancelled" if validation["status"] == "cancelled" else "timeout" if validation["status"] == "timeout" else "failed"
        elif spec.validation_command is not None:
            validation = {"status": "skipped", "reason": "measurement did not complete"}
    except KeyboardInterrupt:
        state, error = "cancelled", "interrupted by user"
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        state, error = "failed", str(exc)
    finally:
        after = {label: repository_identity(repo, output_root) for label, repo in repos.items()}
        _write_json(run_dir / "repositories.final.json", after)
        _write_json(run_dir / "source_checks.json", source_checks)
        identities = list(before.values()) + list(after.values())
        identities.extend(identity for check in source_checks for identity in (check["before"], check["after"]))
        known = bool(before) and all(item.get("available") for item in identities)
        unchanged = (known and all(before[label]["snapshot_hash"] == after[label]["snapshot_hash"] for label in repos)
                     and all(check["before"]["snapshot_hash"] == check["after"]["snapshot_hash"] ==
                             before[check["side"]]["snapshot_hash"] for check in source_checks))
        integrity = {"verified": bool(unchanged), "reason": "Git source snapshots unchanged" if unchanged else
                     "Git source snapshot unavailable" if not known else "source changed during experiment",
                     "command_checks": len(source_checks)}
        result = evaluate(spec, baseline, candidate, validation, integrity)
        if error:
            result["reasons"].append(error)
            if state == "failed":
                result["verdict"] = "FAIL"
        _write_json(run_dir / "evaluation.json", result)
        event("complete", state=state, verdict=result["verdict"])
        _write_json(run_dir / "status.json", {"state": state, "phase": "complete", "verdict": result["verdict"],
                                             "run_dir": str(run_dir), "elapsed_seconds": time.monotonic() - started,
                                             "error": error})
        write_report(run_dir)
    return run_dir
