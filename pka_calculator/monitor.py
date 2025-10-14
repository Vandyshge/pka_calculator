import pandas as pd
import subprocess
from pathlib import Path


def get_squeue_output():
    try:
        result = subprocess.run(["squeue"], capture_output=True, text=True, check=True)
        return result.stdout
    except Exception as e:
        print(f"Error when executing squeue: {e}")
        return ""


def parse_squeue(output, user):
    jobs = []
    for line in output.splitlines():
        if not line.strip() or line.startswith("JOBID"):
            continue
        parts = line.split()
        if len(parts) < 7:
            continue
        job_user = parts[3]
        status = parts[4]
        if job_user == user and status in ("R", "PD"):
            job = {
                "JOBID": parts[0],
                "PARTITION": parts[1],
                "NAME": parts[2],
                "USER": job_user,
                "ST": status,
                "TIME": parts[5],
                "NODES": parts[6],
            }
            # если Pending — причина идёт в скобках
            if status == "PD":
                job["NODELIST"] = " ".join(parts[7:]) if len(parts) > 7 else ""
            else:
                job["NODELIST"] = parts[7] if len(parts) > 7 else ""
            jobs.append(job)
    return jobs


def load_all_summaries(summary_path):
    path = Path(summary_path)
    all_files = list(path.glob("calculations_summary_*.csv"))
    if not all_files:
        print(f"No files found calculations_summary_*.csv в {summary_path}")
        return pd.DataFrame()

    dfs = []
    for f in all_files:
        try:
            df = pd.read_csv(f, sep=r"\s*[,;]\s*|\t", engine="python")
            dfs.append(df)
        except Exception as e:
            print(f"Error while reading {f}: {e}")
    if dfs:
        return pd.concat(dfs, ignore_index=True)
    return pd.DataFrame()


def analyze_jobs(squeue_jobs, summary_df):
    running_jobs = []
    for job in squeue_jobs:
        tasks = summary_df[summary_df["Job ID"].astype(str) == job["JOBID"]]
        if not tasks.empty:
            for _, row in tasks.iterrows():
                running_jobs.append(
                    {
                        "Job ID": job["JOBID"],
                        "Molecule": row.get("Molecule", ""),
                        "Method": row.get("Method", ""),
                        "Form": row.get("Form", ""),
                        "Status": job["ST"],
                        "Node": job["NODELIST"],
                        "Time": job["TIME"],
                    }
                )
    return running_jobs


def monitor_jobs(summary_path, user):
    """Displays a list of the user's current tasks (R и PD)."""
    squeue_output = get_squeue_output()
    squeue_jobs = parse_squeue(squeue_output, user)
    summary_df = load_all_summaries(summary_path)

    if summary_df.empty:
        print("No task data available")
        return

    running_jobs = analyze_jobs(squeue_jobs, summary_df)

    if not running_jobs:
        print("There are no running or pending tasks")
        return

    print("\nCurrent jobs:")
    print("-" * 92)
    print(
        f"{'Job ID':<10}{'Molecule':<15}{'Method':<12}{'Form':<20}{'Status':<8}{'Node':<15}{'Time':<10}"
    )
    print("-" * 92)

    for job in running_jobs:
        print(
            f"{job['Job ID']:<10}{str(job['Molecule']):<15}{job['Method']:<12}"
            f"{job['Form']:<20}{job['Status']:<8}{job['Node']:<15}{job['Time']:<10}"
        )

    total = len(running_jobs)
    nR = sum(1 for j in running_jobs if j["Status"] == "R")
    nPD = sum(1 for j in running_jobs if j["Status"] == "PD")
    print(f"\nSummary: {total} tasks ({nR} running, {nPD} pending)")