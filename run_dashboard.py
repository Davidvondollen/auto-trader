#!/usr/bin/env python3
"""
Dashboard Launcher
Starts the Streamlit web dashboard for the trading system.
"""

import subprocess
import sys
from pathlib import Path


def main():
    """Launch the Streamlit dashboard."""
    dashboard_path = Path(__file__).parent / "src" / "visualization" / "dashboard.py"

    if not dashboard_path.exists():
        print(f"Error: Dashboard file not found at {dashboard_path}")
        sys.exit(1)

    print("=" * 60)
    print("Agentic Trading System - Web Dashboard")
    print("=" * 60)
    print("\nStarting dashboard...")
    print(f"Dashboard will open in your browser at http://localhost:8501")
    print("\nPress Ctrl+C to stop the dashboard\n")

    # Launch Streamlit
    try:
        subprocess.run([
            sys.executable,
            "-m",
            "streamlit",
            "run",
            str(dashboard_path),
            "--server.port=8501",
            "--server.headless=true",
            "--browser.gatherUsageStats=false"
        ])
    except KeyboardInterrupt:
        print("\n\nDashboard stopped by user")
    except Exception as e:
        print(f"\nError starting dashboard: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
