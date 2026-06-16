"""
QuantBet-EV: Windows Task Scheduler Setup
One-click setup for Windows scheduled task, daily automated operation of the quant system
"""

import subprocess
import os
import sys


TASK_NAME = "QuantBetEV_DailyRun"
RUN_TIME = "09:00"  # Run daily at 09:00


def get_python_path():
    """Get current Python interpreter path"""
    return sys.executable


def get_main_path():
    """Get absolute path to main.py"""
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base, 'main.py')


def get_project_dir():
    """Get project root directory"""
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def create_daily_task(time_str=RUN_TIME, source='mock', bankroll=10000, kelly=0.25):
    """
    Create Windows scheduled task
    Use .bat wrapper script to avoid schtasks command line length limit
    """
    project_dir = get_project_dir()
    bat_path = os.path.join(project_dir, 'run_daily.bat')
    log_dir = os.path.join(project_dir, 'logs')
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, 'daily_run.log')

    # Verify .bat file exists
    if not os.path.exists(bat_path):
        print(f"[!] Bat file not found: {bat_path}")
        return False

    # Delete existing task with the same name first
    delete_task()

    # Create new task - call .bat directly
    cmd = [
        'schtasks', '/Create',
        '/TN', TASK_NAME,
        '/TR', bat_path,
        '/SC', 'DAILY',
        '/ST', time_str,
        '/F'
    ]

    print(f"Creating Windows scheduled task: {TASK_NAME}")
    print(f"  Schedule: Daily at {time_str}")
    print(f"  Script: {bat_path}")
    print(f"  Log file: {log_file}")

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print(f"\n[+] Task created successfully!")
        print(f"    Task Name: {TASK_NAME}")
        print(f"    Log file: {log_file}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n[!] Failed to create task: {e}")
        print(f"    stdout: {e.stdout}")
        print(f"    stderr: {e.stderr}")
        return False


def delete_task():
    """Delete the task"""
    cmd = ['schtasks', '/Delete', '/TN', TASK_NAME, '/F']
    try:
        subprocess.run(cmd, capture_output=True, text=True, check=True)
        print(f"  Removed existing task: {TASK_NAME}")
    except subprocess.CalledProcessError:
        pass  # task does not exist


def show_task():
    """Query task status"""
    cmd = ['schtasks', '/Query', '/TN', TASK_NAME]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print("\n" + "="*60)
        print("Task Status:")
        print("="*60)
        print(result.stdout)
    except subprocess.CalledProcessError:
        print(f"[!] Task '{TASK_NAME}' not found.")


def run_once():
    """Run once for testing"""
    python_exe = get_python_path()
    main_py = get_main_path()
    project_dir = get_project_dir()
    log_dir = os.path.join(project_dir, 'logs')
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, 'manual_run.log')

    print(f"Running main.py once...")
    cmd = f'cd /d "{project_dir}" && "{python_exe}" "{main_py}" --source api --bankroll 10000 --kelly 0.25'

    result = subprocess.run(
        f'cmd /c "{cmd}"',
        capture_output=True, text=True, shell=True
    )
    print(result.stdout)
    if result.stderr:
        print("STDERR:", result.stderr)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='QuantBet-EV Windows Task Scheduler')
    parser.add_argument('action', choices=['create', 'delete', 'show', 'run'], help='Action to perform')
    parser.add_argument('--time', type=str, default=RUN_TIME, help='Daily run time (HH:MM)')
    parser.add_argument('--source', type=str, default='api', choices=['mock', 'api'], help='Data source')

    args = parser.parse_args()

    if args.action == 'create':
        create_daily_task(time_str=args.time, source=args.source)
    elif args.action == 'delete':
        delete_task()
        print(f"[+] Task '{TASK_NAME}' deleted.")
    elif args.action == 'show':
        show_task()
    elif args.action == 'run':
        run_once()