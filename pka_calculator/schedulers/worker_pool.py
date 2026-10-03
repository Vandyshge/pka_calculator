from __future__ import annotations

"""Standalone dynamic worker pool for grouped ORCA calculations.

This module intentionally depends only on the Python standard library so that
``prepare_batch_scripts`` can copy it into every Slurm batch directory and run
it directly on a compute node.

The design was validated on the target cluster with ORCA 6.1.1 + OpenMPI 4.1.1:

* one Slurm allocation exposes MPI slots with ``--ntasks=N``;
* ORCA is called directly (never through ``srun`` or external ``mpirun``);
* each worker owns a disjoint set of physical cores;
* ``taskset`` constrains the ORCA driver;
* ``OMPI_MCA_hwloc_base_cpu_list`` confines ORCA's internally launched MPI
  modules to the worker's physical-core set;
* a worker immediately takes the next pending calculation when it becomes free.
"""

import argparse
import csv
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TextIO

NORMAL_MARKER = "ORCA TERMINATED NORMALLY"

EVENT_FIELDS = [
    "time",
    "event",
    "slot",
    "cpus",
    "hwloc_cores",
    "calc_id",
    "pid",
    "return_code",
    "normal_termination",
    "duration_s",
    "allowed_logical_cpus",
    "selected_physical_cpus",
    "slots",
    "pending",
    "successes",
    "failures",
]


def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def read_cpu_core(cpu: int) -> tuple[int, int]:
    """Return ``(socket, physical_core_id)`` for a Linux logical CPU."""
    base = Path(f"/sys/devices/system/cpu/cpu{cpu}/topology")
    core = int((base / "core_id").read_text().strip())
    package = int((base / "physical_package_id").read_text().strip())
    return package, core


def os_cpu_to_hwloc_core_id(cpu: int) -> int:
    """Convert an OS CPU id to the hwloc logical Core id used by OpenMPI 4.1."""
    executable = shutil.which("hwloc-calc")
    if executable is None:
        raise RuntimeError(
            "hwloc-calc was not found in PATH. Load the OpenMPI/hwloc module "
            "before starting the worker pool."
        )
    result = subprocess.run(
        [
            executable,
            "-I",
            "core",
            "--physical-input",
            "--logical-output",
            f"pu:{cpu}",
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"hwloc-calc failed for OS CPU {cpu}: "
            + (result.stderr or result.stdout).strip()
        )
    value = result.stdout.strip()
    if not re.fullmatch(r"\d+", value):
        raise RuntimeError(f"Unexpected hwloc-calc output for OS CPU {cpu}: {value!r}")
    return int(value)


def compress_cpu_list(cpus: list[int]) -> str:
    """Return a taskset-compatible comma-separated CPU list."""
    return ",".join(str(cpu) for cpu in cpus)


def parse_linux_cpu_list(value: str) -> set[int]:
    """Parse Linux ``Cpus_allowed_list`` syntax, e.g. ``34-37,98-101``."""
    cpus: set[int] = set()
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start, end = part.split("-", 1)
            cpus.update(range(int(start), int(end) + 1))
        else:
            cpus.add(int(part))
    return cpus


def output_is_success(path: Path) -> bool:
    """ORCA shell exit codes are not reliable; require the normal marker."""
    if not path.exists():
        return False
    try:
        return NORMAL_MARKER in path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return False


@dataclass
class Slot:
    slot_id: int
    cpus: list[int]
    hwloc_core_ids: list[int]
    calc: dict[str, str] | None = None
    proc: subprocess.Popen | None = None
    stdout: TextIO | None = None
    stderr: TextIO | None = None
    started_monotonic: float | None = None
    started_iso: str | None = None

    @property
    def free(self) -> bool:
        return self.proc is None


class Dispatcher:
    def __init__(
        self,
        *,
        root: str | Path,
        queue_file: str | Path,
        orca_exe: str,
        cores_per_calc: int,
        max_workers: int,
        poll_s: float = 2.0,
        preflight: bool = True,
    ) -> None:
        self.root = Path(root).expanduser().resolve()
        self.queue_file = Path(queue_file).expanduser().resolve()
        self.orca_exe = str(orca_exe)
        self.cores_per_calc = int(cores_per_calc)
        self.max_workers = int(max_workers)
        self.poll_s = float(poll_s)
        self.preflight = bool(preflight)
        self.stop_requested = False

        if self.cores_per_calc < 1:
            raise ValueError("cores_per_calc must be >= 1")
        if self.max_workers < 1:
            raise ValueError("max_workers must be >= 1")
        if self.poll_s <= 0:
            raise ValueError("poll_s must be > 0")

        self.events_path = self.root / "events.csv"
        self.status_path = self.root / "status.csv"
        self.summary_path = self.root / "summary.txt"
        self.topology_path = self.root / "cpu_topology.json"

        self.queue = self._load_queue()
        self.pending: list[dict[str, str]] = []
        self.skipped: list[dict[str, str]] = []
        self.finished: list[dict[str, object]] = []

        self.allowed_logical_cpus = sorted(os.sched_getaffinity(0))
        physical_map = self._select_one_cpu_per_physical_core()
        required_physical = self.max_workers * self.cores_per_calc
        if len(physical_map) < required_physical:
            raise RuntimeError(
                f"Worker pool needs {required_physical} physical cores "
                f"({self.max_workers} workers x {self.cores_per_calc}), but the Slurm "
                f"affinity contains only {len(physical_map)} physical cores."
            )

        # The Slurm allocation may expose more CPUs than requested (for example
        # with --exclusive). Never create more workers than the batch planner intended.
        self.physical_cpu_map = physical_map[:required_physical]
        self.allowed_cpus = [item["cpu"] for item in self.physical_cpu_map]
        self.slots = self._make_slots()
        self._write_cpu_topology_snapshot()

        for calc in self.queue:
            output = self._workdir(calc) / "output.out"
            if output_is_success(output):
                self.skipped.append({**calc, "status": "SKIPPED_SUCCESS"})
            else:
                self.pending.append(calc)

        signal.signal(signal.SIGTERM, self._handle_stop)
        signal.signal(signal.SIGINT, self._handle_stop)

    def _load_queue(self) -> list[dict[str, str]]:
        with self.queue_file.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        if not rows:
            raise RuntimeError(f"Queue is empty: {self.queue_file}")
        required = {"calc_id", "workdir"}
        missing = required - set(rows[0])
        if missing:
            raise RuntimeError(f"Queue is missing required columns: {sorted(missing)}")
        return rows

    def _workdir(self, calc: dict[str, str]) -> Path:
        path = Path(calc["workdir"]).expanduser()
        return path.resolve() if path.is_absolute() else (self.root / path).resolve()

    def _select_one_cpu_per_physical_core(self) -> list[dict[str, object]]:
        groups: dict[tuple[int, int], list[int]] = {}
        for cpu in self.allowed_logical_cpus:
            groups.setdefault(read_cpu_core(cpu), []).append(cpu)

        selected: list[dict[str, object]] = []
        for (socket, core), logicals in sorted(groups.items(), key=lambda item: item[0]):
            siblings = sorted(logicals)
            selected.append(
                {
                    "socket": socket,
                    "core": core,
                    "cpu": siblings[0],
                    "siblings_visible": siblings,
                }
            )
        return selected

    def _make_slots(self) -> list[Slot]:
        hwloc_map = {cpu: os_cpu_to_hwloc_core_id(cpu) for cpu in self.allowed_cpus}
        slots: list[Slot] = []
        for slot_id in range(self.max_workers):
            start = slot_id * self.cores_per_calc
            cpus = self.allowed_cpus[start : start + self.cores_per_calc]
            hwloc_ids = [hwloc_map[cpu] for cpu in cpus]
            if len(set(hwloc_ids)) != len(hwloc_ids):
                raise RuntimeError(
                    f"Worker slot {slot_id} maps multiple OS CPUs to the same hwloc core: "
                    f"cpus={cpus}, hwloc={hwloc_ids}"
                )
            slots.append(Slot(slot_id=slot_id, cpus=cpus, hwloc_core_ids=hwloc_ids))
        return slots

    def _write_cpu_topology_snapshot(self) -> None:
        topology: dict[str, object] = {
            "host": os.uname().nodename,
            "created_at": now_iso(),
            "allowed_logical_cpus": self.allowed_logical_cpus,
            "selected_physical_cpus": self.allowed_cpus,
            "cpus": {},
            "slots": [],
        }
        cpu_records = topology["cpus"]
        assert isinstance(cpu_records, dict)
        for cpu in self.allowed_logical_cpus:
            socket, core = read_cpu_core(cpu)
            cpu_records[str(cpu)] = {"socket": socket, "core": core}

        slot_records = topology["slots"]
        assert isinstance(slot_records, list)
        for slot in self.slots:
            slot_records.append(
                {
                    "slot_id": slot.slot_id,
                    "os_cpus": slot.cpus,
                    "hwloc_core_ids": slot.hwloc_core_ids,
                    "physical_cores": [
                        {"socket": read_cpu_core(cpu)[0], "core": read_cpu_core(cpu)[1]}
                        for cpu in slot.cpus
                    ],
                }
            )
        self.topology_path.write_text(
            json.dumps(topology, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )

    def _handle_stop(self, signum, _frame) -> None:
        print(
            f"\nReceived signal {signum}; stopping queue and terminating active ORCA jobs.",
            flush=True,
        )
        self.stop_requested = True

    def _event(
        self,
        event: str,
        slot: Slot | None = None,
        calc: dict[str, str] | None = None,
        **extra: object,
    ) -> None:
        row: dict[str, object] = {key: "" for key in EVENT_FIELDS}
        row.update(
            {
                "time": now_iso(),
                "event": event,
                "slot": "" if slot is None else slot.slot_id,
                "cpus": "" if slot is None else compress_cpu_list(slot.cpus),
                "hwloc_cores": ""
                if slot is None
                else ",".join(map(str, slot.hwloc_core_ids)),
                "calc_id": "" if calc is None else calc["calc_id"],
            }
        )
        for key, value in extra.items():
            if key in row:
                row[key] = value

        exists = self.events_path.exists() and self.events_path.stat().st_size > 0
        with self.events_path.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=EVENT_FIELDS, extrasaction="ignore")
            if not exists:
                writer.writeheader()
            writer.writerow(row)

    def _preflight_slot(self, slot: Slot) -> None:
        """Verify that MPI ranks cannot escape this worker's physical cores."""
        driver_cpu_list = compress_cpu_list(slot.cpus)
        core_list = ",".join(map(str, slot.hwloc_core_ids))
        path = self.root / f"preflight_slot_{slot.slot_id}.txt"

        env = os.environ.copy()
        env["OMPI_MCA_hwloc_base_cpu_list"] = core_list
        env["OMPI_MCA_hwloc_base_report_bindings"] = "1"

        probe = (
            'printf "rank=%s pid=%s " "${OMPI_COMM_WORLD_RANK:-?}" "$$"; '
            'grep "^Cpus_allowed_list:" /proc/self/status'
        )
        result = subprocess.run(
            [
                "taskset",
                "-c",
                driver_cpu_list,
                "mpirun",
                "-np",
                str(self.cores_per_calc),
                "bash",
                "-c",
                probe,
            ],
            cwd=self.root,
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )

        target_physical = {read_cpu_core(cpu) for cpu in slot.cpus}
        path.write_text(
            f"slot={slot.slot_id}\n"
            f"driver_os_cpus={driver_cpu_list}\n"
            f"hwloc_core_list={core_list}\n"
            f"target_physical_cores={sorted(target_physical)}\n"
            f"return_code={result.returncode}\n"
            + result.stdout,
            encoding="utf-8",
        )
        if result.returncode != 0:
            raise RuntimeError(f"OpenMPI preflight failed for slot {slot.slot_id}. See {path}")

        # Output from several ranks may interleave on one line; parse globally.
        masks = re.findall(r"Cpus_allowed_list:\s*([0-9,-]+)", result.stdout)
        if len(masks) != self.cores_per_calc:
            raise RuntimeError(
                f"OpenMPI preflight slot {slot.slot_id}: expected {self.cores_per_calc} "
                f"rank affinity masks, got {len(masks)}. See {path}"
            )

        for mask in masks:
            linux_cpus = parse_linux_cpu_list(mask)
            physical = {read_cpu_core(cpu) for cpu in linux_cpus}
            escaped = physical - target_physical
            if escaped:
                raise RuntimeError(
                    f"OpenMPI preflight slot {slot.slot_id}: rank escaped worker cores; "
                    f"mask={mask}, escaped={sorted(escaped)}. See {path}"
                )

        print(
            f"preflight slot={slot.slot_id} OK: physical={sorted(target_physical)} "
            f"rank_masks={masks}",
            flush=True,
        )

    def _archive_previous_failed_output(self, workdir: Path) -> None:
        stamp = time.strftime("%Y%m%d_%H%M%S")
        output = workdir / "output.out"
        error = workdir / "output.err"
        if output.exists() and not output_is_success(output):
            target = workdir / f"output.failed.{stamp}.out"
            counter = 1
            while target.exists():
                target = workdir / f"output.failed.{stamp}.{counter}.out"
                counter += 1
            output.rename(target)
        if error.exists():
            target = workdir / f"output.previous.{stamp}.err"
            counter = 1
            while target.exists():
                target = workdir / f"output.previous.{stamp}.{counter}.err"
                counter += 1
            error.rename(target)

    def _launch(self, slot: Slot, calc: dict[str, str]) -> None:
        workdir = self._workdir(calc)
        input_file = calc.get("input_file", "input.inp") or "input.inp"
        workdir.mkdir(parents=True, exist_ok=True)
        self._archive_previous_failed_output(workdir)

        stdout = (workdir / "output.out").open("w", encoding="utf-8")
        stderr = (workdir / "output.err").open("w", encoding="utf-8")
        cpu_list = compress_cpu_list(slot.cpus)
        hwloc_list = ",".join(map(str, slot.hwloc_core_ids))

        (workdir / "node.txt").write_text(os.uname().nodename + "\n", encoding="utf-8")
        (workdir / "job_id.txt").write_text(
            os.environ.get("SLURM_JOB_ID", "") + "\n", encoding="utf-8"
        )
        (workdir / "worker.txt").write_text(
            f"worker_slot={slot.slot_id} os_cpus={cpu_list} hwloc_cores={hwloc_list}\n"
            f"OMPI_MCA_hwloc_base_cpu_list={hwloc_list}\n",
            encoding="utf-8",
        )

        env = os.environ.copy()
        env["OMPI_MCA_hwloc_base_cpu_list"] = hwloc_list
        env["OMPI_MCA_hwloc_base_report_bindings"] = "1"

        # ORCA must be called directly. It launches its own mpirun for parallel modules.
        cmd = [
            "taskset",
            "-c",
            cpu_list,
            self.orca_exe,
            input_file,
        ]
        proc = subprocess.Popen(
            cmd,
            cwd=workdir,
            stdout=stdout,
            stderr=stderr,
            env=env,
            start_new_session=True,
        )

        slot.calc = calc
        slot.proc = proc
        slot.stdout = stdout
        slot.stderr = stderr
        slot.started_monotonic = time.monotonic()
        slot.started_iso = now_iso()

        self._event("START", slot, calc, pid=proc.pid)
        print(
            f"[{now_iso()}] START slot={slot.slot_id} cpus={cpu_list} "
            f"calc={calc['calc_id']} pid={proc.pid}",
            flush=True,
        )

    def _finish_slot(self, slot: Slot) -> bool:
        assert slot.proc is not None and slot.calc is not None
        return_code = slot.proc.poll()
        if return_code is None:
            return False

        duration = time.monotonic() - (slot.started_monotonic or time.monotonic())
        calc = slot.calc
        workdir = self._workdir(calc)
        normal = output_is_success(workdir / "output.out")
        status = "SUCCESS" if normal else "FAILED"

        if slot.stdout is not None:
            slot.stdout.close()
        if slot.stderr is not None:
            slot.stderr.close()

        (workdir / "exit_code.txt").write_text(f"{return_code}\n", encoding="utf-8")
        (workdir / "wall_seconds.txt").write_text(f"{duration:.6f}\n", encoding="utf-8")

        result: dict[str, object] = {
            **calc,
            "status": status,
            "return_code": return_code,
            "normal_termination": normal,
            "slot": slot.slot_id,
            "cpus": compress_cpu_list(slot.cpus),
            "hwloc_cores": ",".join(map(str, slot.hwloc_core_ids)),
            "started": slot.started_iso,
            "finished": now_iso(),
            "duration_s": round(duration, 3),
        }
        self.finished.append(result)
        self._event(
            "FINISH",
            slot,
            calc,
            return_code=return_code,
            normal_termination=normal,
            duration_s=round(duration, 3),
        )
        print(
            f"[{now_iso()}] FINISH slot={slot.slot_id} calc={calc['calc_id']} "
            f"rc={return_code} normal={normal} duration={duration:.1f}s",
            flush=True,
        )

        slot.calc = None
        slot.proc = None
        slot.stdout = None
        slot.stderr = None
        slot.started_monotonic = None
        slot.started_iso = None
        return True

    def _write_status(self) -> None:
        rows: list[dict[str, object]] = [*self.skipped, *self.finished]
        for slot in self.slots:
            if slot.calc is not None:
                rows.append(
                    {
                        **slot.calc,
                        "status": "RUNNING",
                        "slot": slot.slot_id,
                        "cpus": compress_cpu_list(slot.cpus),
                        "hwloc_cores": ",".join(map(str, slot.hwloc_core_ids)),
                        "started": slot.started_iso,
                    }
                )

        known = {str(row["calc_id"]) for row in rows}
        for calc in self.pending:
            if calc["calc_id"] not in known:
                rows.append({**calc, "status": "PENDING"})

        fields = sorted({key for row in rows for key in row}) if rows else ["calc_id", "status"]
        with self.status_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)

    def _terminate_active(self) -> None:
        for slot in self.slots:
            if slot.proc is None or slot.proc.poll() is not None:
                continue
            print(
                f"Terminating slot={slot.slot_id} calc={slot.calc['calc_id'] if slot.calc else '?'}",
                flush=True,
            )
            try:
                os.killpg(slot.proc.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass

        deadline = time.monotonic() + 10.0
        while time.monotonic() < deadline:
            alive = [
                slot
                for slot in self.slots
                if slot.proc is not None and slot.proc.poll() is None
            ]
            if not alive:
                break
            time.sleep(0.2)

        for slot in self.slots:
            if slot.proc is not None and slot.proc.poll() is None:
                try:
                    os.killpg(slot.proc.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass

    def _write_summary(self) -> tuple[int, int]:
        successes = sum(1 for row in self.finished if row["status"] == "SUCCESS")
        failures = sum(1 for row in self.finished if row["status"] == "FAILED")
        lines = [
            f"queue_total={len(self.queue)}",
            f"skipped_success={len(self.skipped)}",
            f"new_success={successes}",
            f"failed={failures}",
            f"pending={len(self.pending)}",
        ]
        self.summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return successes, failures

    def run(self) -> int:
        print("===== pka-calculator ORCA worker pool =====", flush=True)
        print(f"root={self.root}", flush=True)
        print(f"allowed_logical_cpus={self.allowed_logical_cpus}", flush=True)
        print(f"selected_physical_cpus={self.allowed_cpus}", flush=True)
        print(f"cores_per_calc={self.cores_per_calc}", flush=True)
        print(f"worker_slots={len(self.slots)}", flush=True)
        print(f"queue_total={len(self.queue)}", flush=True)
        print(f"already_successful={len(self.skipped)}", flush=True)
        print(f"pending={len(self.pending)}", flush=True)
        for slot in self.slots:
            print(
                f"slot {slot.slot_id}: OS CPUs {slot.cpus} -> "
                f"hwloc core IDs {slot.hwloc_core_ids}",
                flush=True,
            )

        self._event(
            "DISPATCHER_START",
            allowed_logical_cpus=",".join(map(str, self.allowed_logical_cpus)),
            selected_physical_cpus=",".join(map(str, self.allowed_cpus)),
            slots=len(self.slots),
            pending=len(self.pending),
        )

        # A fully completed resubmitted batch should exit without needing MPI/hwloc.
        if not self.pending:
            self._write_status()
            successes, failures = self._write_summary()
            self._event("DISPATCHER_FINISH", successes=successes, failures=failures)
            return 0

        if self.preflight:
            print("===== OpenMPI worker-isolation preflight =====", flush=True)
            for slot in self.slots:
                self._preflight_slot(slot)

        try:
            while self.pending or any(not slot.free for slot in self.slots):
                if self.stop_requested:
                    break

                # Reap first, then refill every newly free slot immediately.
                for slot in self.slots:
                    if not slot.free:
                        self._finish_slot(slot)

                for slot in self.slots:
                    if self.pending and slot.free and not self.stop_requested:
                        self._launch(slot, self.pending.pop(0))

                self._write_status()
                if self.pending or any(not slot.free for slot in self.slots):
                    time.sleep(self.poll_s)

            if self.stop_requested:
                self._terminate_active()
                for slot in self.slots:
                    if slot.free:
                        continue
                    try:
                        slot.proc.wait(timeout=2)  # type: ignore[union-attr]
                    except Exception:
                        pass
                    self._finish_slot(slot)
                self._write_status()
                self._write_summary()
                self._event("DISPATCHER_STOPPED")
                return 130

            self._write_status()
            successes, failures = self._write_summary()
            self._event("DISPATCHER_FINISH", successes=successes, failures=failures)
            return 1 if failures else 0
        finally:
            # Avoid leaving child processes behind on unexpected Python exceptions.
            if any(slot.proc is not None and slot.proc.poll() is None for slot in self.slots):
                self._terminate_active()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Dynamic ORCA worker-pool dispatcher")
    parser.add_argument("--root", required=True)
    parser.add_argument("--queue", required=True)
    parser.add_argument("--orca", required=True)
    parser.add_argument("--cores-per-calc", required=True, type=int)
    parser.add_argument("--workers", required=True, type=int)
    parser.add_argument("--poll", type=float, default=2.0)
    parser.add_argument("--skip-preflight", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    dispatcher = Dispatcher(
        root=args.root,
        queue_file=args.queue,
        orca_exe=args.orca,
        cores_per_calc=args.cores_per_calc,
        max_workers=args.workers,
        poll_s=args.poll,
        preflight=not args.skip_preflight,
    )
    return dispatcher.run()


if __name__ == "__main__":
    sys.exit(main())
