import pandas as pd
import subprocess
import re
from pathlib import Path

def get_squeue_output():
    try:
        result = subprocess.run(['squeue'], capture_output=True, text=True, check=True)
        return result.stdout
    except:
        print("Error when executing squeue")
        return ""

def parse_squeue(output, user):
    jobs = []
    for line in output.split('\n'):
        if not line.strip() or line.startswith("JOBID"):
            continue
        
        parts = [p for p in line.split(' ') if p]
        if len(parts) >= 6 and parts[3] in user and parts[4] == "R":
            jobs.append({
                'JOBID': parts[0],
                'PARTITION': parts[1],
                'NAME': parts[2],
                'USER': parts[3],
                'ST': parts[4],
                'TIME': parts[5],
                'NODES': parts[6],
                'NODELIST': parts[7] if len(parts) > 7 else ''
            })
    return jobs

def load_calculations_summary(summary_path):
    try:
        return pd.read_csv(f'{summary_path}/calculations_summary.csv', sep=';', engine='python', 
                         skipinitialspace=True, 
                         names=['Molecule', 'Method', 'Form', 'Job ID', 'Status'])
    except Exception as e:
        print(f"Ошибка при загрузке файла {summary_file}: {e}")
        return pd.DataFrame()

def analyze_jobs(squeue_jobs, summary_df):
    running_jobs = []
    for job in squeue_jobs:
        task = summary_df[summary_df['Job ID'].astype(str) == job['JOBID']]
        if not task.empty:
            running_jobs.append({
                'Job ID': job['JOBID'],
                'Molecule': task.iloc[0]['Molecule'],
                'Method': task.iloc[0]['Method'],
                'Form': task.iloc[0]['Form'],
                'Status': job['ST'],
                'Node': job['NODELIST'],
                'Time': job['TIME']
            })
    return running_jobs

def monitor_jobs(summary_path, user):
    """Output of a list of currently running tasks"""
    squeue_output = get_squeue_output()
    squeue_jobs = parse_squeue(squeue_output, user)
    summary_df = load_calculations_summary(summary_path)
    
    running_jobs = analyze_jobs(squeue_jobs, summary_df)

    if not running_jobs:
        print("There are no running tasks")
        return
    
    print("\nCurrent running tasks:")
    print("-" * 84)
    print(f"{'Job ID':<10}{'Molecule':<20}{'Method':<10}{'Form':<15}{'Status':<10}{'Node':<10}{'Time':<10}")
    print("-" * 84)
    
    for job in running_jobs:
        print(f"{job['Job ID']:<10}{job['Molecule']:<20}{job['Method']:<10}"
              f"{job['Form']:<15}{job['Status']:<10}{job['Node']:<10}{job['Time']:<10}")