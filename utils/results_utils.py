"""Utilities for recording per-run results and compute requirements.

This module provides the building blocks used by ``slam.py`` to write a
``results.json`` file into each run's output directory. The payload captures:

* quality metrics (ATE RMSE, PSNR/SSIM/LPIPS),
* compute requirements (per-phase timing/FPS, peak memory, model size),
* an environment fingerprint (GPU, CUDA, PyTorch, CPU, host).

Design notes
------------
* Memory is sampled in a background daemon thread. GPU usage is read
  *device-wide* via ``torch.cuda.mem_get_info`` so it also captures the
  separately-spawned backend process; RAM is read across the whole process
  tree (main process + children) via ``psutil`` when available, with a
  stdlib ``resource`` fallback for the main process.
* ``psutil`` is optional: every code path degrades gracefully if it is not
  installed.
* The JSON is written atomically (temp file + ``os.replace``) so a partial
  write can never corrupt a previous result.
"""

import contextlib
import datetime
import json
import math
import os
import platform
import socket
import sys
import threading
import time

import torch

try:  # psutil is optional.
    import psutil
except Exception:  # pragma: no cover - depends on the environment
    psutil = None


# Number of color-refinement optimization iterations performed by
# ``utils.slam_backend.BackEnd.color_refinement``. Kept here so the timing
# reporter and the backend share a single source of truth.
COLOR_REFINEMENT_ITERATIONS = 26000

RESULTS_FILENAME = "results.json"
RESULTS_SCHEMA_VERSION = 1


# ---------------------------------------------------------------------------
# Environment fingerprint
# ---------------------------------------------------------------------------
def _cpu_model():
    """Best-effort CPU model name (Linux ``/proc/cpuinfo``, else fallback)."""
    try:
        with open("/proc/cpuinfo", "r") as f:
            for line in f:
                if line.lower().startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or None


def collect_environment():
    """Collect a hardware/software fingerprint for reproducibility."""
    env = {
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "os": platform.system(),
        "python_version": platform.python_version(),
        "pytorch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "cudnn_version": None,
        "cpu_model": _cpu_model(),
        "cpu_count_logical": os.cpu_count(),
        "cpu_count_physical": None,
        "ram_total_bytes": None,
        "gpu_name": None,
        "gpu_count": None,
        "gpu_total_vram_bytes": None,
        "gpu_compute_capability": None,
        "psutil_available": psutil is not None,
    }

    try:
        if torch.backends.cudnn.is_available():
            env["cudnn_version"] = torch.backends.cudnn.version()
    except Exception:  # pragma: no cover - backend dependent
        pass

    if psutil is not None:
        try:
            env["cpu_count_physical"] = psutil.cpu_count(logical=False)
            env["ram_total_bytes"] = int(psutil.virtual_memory().total)
        except Exception:  # pragma: no cover - psutil platform quirk
            pass

    if torch.cuda.is_available():
        try:
            props = torch.cuda.get_device_properties(0)
            env["gpu_name"] = props.name
            env["gpu_count"] = torch.cuda.device_count()
            env["gpu_total_vram_bytes"] = int(props.total_memory)
            env["gpu_compute_capability"] = f"{props.major}.{props.minor}"
        except Exception:  # pragma: no cover - driver dependent
            pass

    return env


# ---------------------------------------------------------------------------
# JSON helpers
# ---------------------------------------------------------------------------
def read_json(path):
    """Return parsed JSON at ``path`` or ``None`` if missing/invalid."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None


def parse_run_timestamp(dir_name):
    """Parse a ``%Y-%m-%d-%H-%M-%S`` run directory name into ISO format."""
    try:
        return datetime.datetime.strptime(dir_name, "%Y-%m-%d-%H-%M-%S").isoformat()
    except (ValueError, TypeError):
        return None


def _json_default(obj):
    """Serialize numpy/torch scalars and fall back to ``str``."""
    item = getattr(obj, "item", None)
    if callable(item):
        try:
            return item()
        except Exception:  # pragma: no cover - exotic scalar
            pass
    return str(obj)


def _json_safe(obj):
    """Recursively replace non-finite floats with ``None`` so JSON stays valid."""
    if isinstance(obj, dict):
        return {key: _json_safe(value) for key, value in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(value) for value in obj]
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    return obj


def write_results_json(save_dir, payload, filename=RESULTS_FILENAME):
    """Atomically write ``payload`` as JSON into ``save_dir``; return the path.

    Non-finite floats (e.g. a NaN ATE when alignment is impossible) are written
    as ``null`` so the file stays parseable by strict JSON readers.
    """
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, filename)
    tmp = f"{path}.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(_json_safe(payload), f, indent=2, default=_json_default)
    os.replace(tmp, path)
    return path


# ---------------------------------------------------------------------------
# Payload assembly
# ---------------------------------------------------------------------------
def _resolve_rendering(save_dir, stage, in_memory):
    """Prefer the in-memory rendering result, else read it from disk."""
    if in_memory is not None:
        return in_memory
    return read_json(os.path.join(save_dir, "psnr", stage, "final_result.json"))


def build_results_payload(
    *,
    save_dir,
    config_path,
    config,
    eval_rendering,
    durations,
    total_s,
    fps,
    n_frames,
    num_keyframes,
    ate,
    rendering_before,
    rendering_after,
    gaussians_after_tracking,
    gaussians_final,
    memory,
    refinement_iterations=COLOR_REFINEMENT_ITERATIONS,
):
    """Assemble the full ``results.json`` payload.

    Pure data assembly (no torch objects involved) so it can be unit-tested in
    isolation from the SLAM pipeline.
    """
    # --- quality metrics -------------------------------------------------
    ate_stats = read_json(os.path.join(save_dir, "plot", "stats_final.json")) or {}
    if isinstance(ate, dict):
        for key, value in ate.items():
            ate_stats.setdefault(key, value)
    elif ate is not None:
        ate_stats.setdefault("rmse", float(ate))

    rendering = {
        "before_opt": _resolve_rendering(save_dir, "before_opt", rendering_before),
        "after_opt": _resolve_rendering(save_dir, "after_opt", rendering_after),
    }

    # --- timing ----------------------------------------------------------
    online_s = durations.get("online")
    refinement_s = durations.get("color_refinement")
    phases = {name: {"wall_s": round(value, 4)} for name, value in durations.items()}
    if online_s:
        phases["online"].update({"num_frames": n_frames, "fps": n_frames / online_s})
    if refinement_s:
        phases["color_refinement"].update(
            {
                "iterations": refinement_iterations,
                "iters_per_s": refinement_iterations / refinement_s,
            }
        )
    for stage in ("eval_before_opt", "eval_after_opt"):
        if durations.get(stage):
            phases[stage].update(
                {"num_frames": n_frames, "fps": n_frames / durations[stage]}
            )

    core_s = None
    if online_s is not None or refinement_s is not None:
        core_s = (online_s or 0.0) + (refinement_s or 0.0)

    timing = {
        "total_s": round(total_s, 4),
        "tracked_sum_s": round(sum(durations.values()), 4),
        "slam_online_plus_refinement_s": (
            round(core_s, 4) if core_s is not None else None
        ),
        "overall_fps": fps,
        "phases": phases,
    }

    # --- metadata --------------------------------------------------------
    ds = config.get("Dataset", {})
    calib = ds.get("Calibration", {})
    train = config.get("Training", {})
    dir_name = os.path.basename(os.path.normpath(save_dir))

    return {
        "schema_version": RESULTS_SCHEMA_VERSION,
        "run": {
            "dir_name": dir_name,
            "path": save_dir,
            "timestamp": parse_run_timestamp(dir_name),
            "saved_at": datetime.datetime.now().isoformat(timespec="seconds"),
            "duration_s": round(total_s, 4),
            "config_path": config_path,
        },
        "dataset": {
            "path": ds.get("dataset_path"),
            "type": ds.get("type"),
            "sensor_type": ds.get("sensor_type"),
            "num_frames": n_frames,
            "width": calib.get("width"),
            "height": calib.get("height"),
        },
        "experiment": {
            "config_path": config_path,
            "lambda_iso": config.get("Experiment", {}).get("lambda_iso"),
            "force_isotropic": config.get("shape", {}).get("force_isotropic"),
            "window_size": train.get("window_size"),
            "mapping_itr_num": train.get("mapping_itr_num"),
            "spherical_harmonics": train.get("spherical_harmonics"),
            "eval_rendering": eval_rendering,
        },
        "environment": collect_environment(),
        "metrics": {"ate": ate_stats, "rendering": rendering},
        "compute": {
            "timing": timing,
            "memory": memory,
            "model": {
                "num_gaussians_after_tracking": gaussians_after_tracking,
                "num_gaussians_final": gaussians_final,
                "num_keyframes": num_keyframes,
                "num_frames": n_frames,
            },
        },
    }


# ---------------------------------------------------------------------------
# Phase timing
# ---------------------------------------------------------------------------
class PhaseTimer:
    """Wall-clock timer that records durations for named phases.

    Use :meth:`phase` as a context manager for a self-contained span, or
    :meth:`lap` to close out the span since the previous mark (handy when the
    region of interest cannot be neatly indented).
    """

    def __init__(self):
        self._t_start = time.perf_counter()
        self._t_last = self._t_start
        self.durations = {}

    def lap(self, name):
        """Record the elapsed time since the previous mark under ``name``."""
        now = time.perf_counter()
        self.durations[name] = now - self._t_last
        self._t_last = now
        return self.durations[name]

    @contextlib.contextmanager
    def phase(self, name):
        """Context manager recording the enclosed span as ``name``."""
        t0 = time.perf_counter()
        try:
            yield
        finally:
            self.durations[name] = time.perf_counter() - t0

    def total(self):
        """Wall-clock seconds since this timer was created."""
        return time.perf_counter() - self._t_start


# ---------------------------------------------------------------------------
# Resource sampling
# ---------------------------------------------------------------------------
def _rusage_maxrss_bytes(children=False):
    """Peak RSS from stdlib ``resource`` (main process or children).

    ``ru_maxrss`` is in KiB on Linux and bytes on macOS.
    """
    try:
        import resource

        which = resource.RUSAGE_CHILDREN if children else resource.RUSAGE_SELF
        raw = resource.getrusage(which).ru_maxrss
    except Exception:
        return None
    if sys.platform == "darwin":
        return int(raw)
    return int(raw) * 1024


def _process_tree_rss_bytes(pid):
    """Summed RSS (bytes) of ``pid`` and all descendants via psutil."""
    if psutil is None:
        return None
    try:
        proc = psutil.Process(pid)
        total = proc.memory_info().rss
        for child in proc.children(recursive=True):
            try:
                total += child.memory_info().rss
            except psutil.Error:
                continue
        return int(total)
    except Exception:
        return None


class ResourceSampler:
    """Background peak sampler for GPU and RAM usage.

    Call :meth:`start` before the measured work, :meth:`stop` afterwards, and
    :meth:`summary` to obtain the results.
    """

    def __init__(self, interval_s=0.5):
        self.interval_s = interval_s
        self._stop_event = threading.Event()
        self._thread = None
        self._pid = os.getpid()
        self._peak_gpu_used = 0
        self._gpu_total = None
        self._peak_ram_tree = None

    # -- lifecycle ---------------------------------------------------------
    def start(self):
        if torch.cuda.is_available():
            try:
                torch.cuda.reset_peak_memory_stats()
                self._gpu_total = int(torch.cuda.mem_get_info()[1])
            except Exception:  # pragma: no cover - driver dependent
                self._gpu_total = None
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run, name="resource-sampler", daemon=True
        )
        self._thread.start()
        return self

    def stop(self):
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=self.interval_s * 3 + 1)
            self._thread = None
        self._sample()  # capture a final reading
        return self

    # -- sampling ----------------------------------------------------------
    def _sample(self):
        if torch.cuda.is_available():
            try:
                free, total = torch.cuda.mem_get_info()
                used = int(total) - int(free)
                if used > self._peak_gpu_used:
                    self._peak_gpu_used = used
                if self._gpu_total is None:
                    self._gpu_total = int(total)
            except Exception:  # pragma: no cover - driver dependent
                pass

        ram = _process_tree_rss_bytes(self._pid)
        if ram is not None and (self._peak_ram_tree is None or ram > self._peak_ram_tree):
            self._peak_ram_tree = ram

    def _run(self):
        while not self._stop_event.wait(self.interval_s):
            self._sample()

    # -- results -----------------------------------------------------------
    def summary(self):
        peak_alloc = peak_reserved = None
        if torch.cuda.is_available():
            try:
                peak_alloc = int(torch.cuda.max_memory_allocated())
                peak_reserved = int(torch.cuda.max_memory_reserved())
            except Exception:  # pragma: no cover - driver dependent
                pass

        fraction = None
        if self._gpu_total and self._peak_gpu_used:
            fraction = self._peak_gpu_used / self._gpu_total

        return {
            "sampling_interval_s": self.interval_s,
            "psutil_available": psutil is not None,
            "peak_gpu_used_bytes": self._peak_gpu_used or None,
            "peak_gpu_total_bytes": self._gpu_total,
            "peak_gpu_used_fraction": fraction,
            "peak_ram_process_tree_bytes": self._peak_ram_tree,
            "peak_ram_main_process_bytes": _rusage_maxrss_bytes(children=False),
            "peak_ram_children_bytes": _rusage_maxrss_bytes(children=True),
            "peak_main_process_cuda_allocated_bytes": peak_alloc,
            "peak_main_process_cuda_reserved_bytes": peak_reserved,
        }


# ---------------------------------------------------------------------------
# Model size
# ---------------------------------------------------------------------------
def gaussian_count(gaussians):
    """Number of Gaussians in a ``GaussianModel`` (or ``None``)."""
    if gaussians is None:
        return None
    try:
        return int(gaussians.get_xyz.shape[0])
    except Exception:
        return None
