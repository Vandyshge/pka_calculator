from __future__ import annotations

import csv
import os
from pathlib import Path

import pytest

from pka_calculator.schedulers import worker_pool


def test_parse_linux_cpu_list():
    assert worker_pool.parse_linux_cpu_list("34-37,98-101") == {
        34, 35, 36, 37, 98, 99, 100, 101
    }


def test_output_success_requires_normal_marker(tmp_path):
    out = tmp_path / "output.out"
    out.write_text("ORCA finished by error termination\n", encoding="utf-8")
    assert not worker_pool.output_is_success(out)
    out.write_text("****ORCA TERMINATED NORMALLY****\n", encoding="utf-8")
    assert worker_pool.output_is_success(out)


def test_dispatcher_dynamic_queue_and_resume(tmp_path, monkeypatch):
    allowed = sorted(os.sched_getaffinity(0))
    if len(allowed) < 2:
        pytest.skip("Need at least two CPUs for taskset worker-pool test")
    selected = allowed[:2]

    # Treat the two actual allowed Linux CPUs as two distinct physical cores.
    topology = {cpu: (0, i) for i, cpu in enumerate(selected)}
    monkeypatch.setattr(worker_pool.os, "sched_getaffinity", lambda _pid: set(selected))
    monkeypatch.setattr(worker_pool, "read_cpu_core", lambda cpu: topology[cpu])
    monkeypatch.setattr(worker_pool, "os_cpu_to_hwloc_core_id", lambda cpu: selected.index(cpu))

    fake_orca = tmp_path / "fake_orca.py"
    fake_orca.write_text(
        "#!/usr/bin/env python3\n"
        "import pathlib, time\n"
        "delay = float(pathlib.Path('input.inp').read_text().strip())\n"
        "time.sleep(delay)\n"
        "print('****ORCA TERMINATED NORMALLY****')\n",
        encoding="utf-8",
    )
    fake_orca.chmod(0o755)

    rows = []
    for i, delay in enumerate([0.12, 0.28, 0.12, 0.12], start=1):
        workdir = tmp_path / f"calc_{i}"
        workdir.mkdir()
        (workdir / "input.inp").write_text(str(delay), encoding="utf-8")
        rows.append({"calc_id": f"c{i}", "workdir": str(workdir), "input_file": "input.inp"})

    queue = tmp_path / "queue.csv"
    with queue.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["calc_id", "workdir", "input_file"])
        writer.writeheader()
        writer.writerows(rows)

    dispatcher = worker_pool.Dispatcher(
        root=tmp_path,
        queue_file=queue,
        orca_exe=str(fake_orca),
        cores_per_calc=1,
        max_workers=2,
        poll_s=0.02,
        preflight=False,
    )
    assert dispatcher.run() == 0
    assert all(worker_pool.output_is_success(Path(row["workdir"]) / "output.out") for row in rows)

    events = list(csv.DictReader((tmp_path / "events.csv").open(encoding="utf-8")))
    starts = [row for row in events if row["event"] == "START"]
    finishes = [row for row in events if row["event"] == "FINISH"]
    assert len(starts) == 4
    assert len(finishes) == 4

    # Run again: all calculations must be skipped, with no ORCA launches.
    second = worker_pool.Dispatcher(
        root=tmp_path,
        queue_file=queue,
        orca_exe=str(fake_orca),
        cores_per_calc=1,
        max_workers=2,
        poll_s=0.02,
        preflight=False,
    )
    assert second.run() == 0
    summary = (tmp_path / "summary.txt").read_text(encoding="utf-8")
    assert "skipped_success=4" in summary
    assert "new_success=0" in summary
