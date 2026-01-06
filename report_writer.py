# report_writer.py

from pathlib import Path
from datetime import datetime
import pandas as pd

REPORT_PATH = Path("reliability_report.html")


def start_report(title: str = "Resting-State Reliability Report") -> None:
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write(
            """
<html>
<head>
<style>
    body { font-family: Arial, sans-serif; margin: 25px; }
    h1 { color: #0b3d91; }
    h2 { color: #333333; margin-top: 30px; }
    table {
        border-collapse: collapse;
        width: 100%;
        margin-top: 10px;
        margin-bottom: 20px;
        font-size: 0.9em;
    }
    table, th, td {
        border: 1px solid #dddddd;
    }
    th, td {
        padding: 6px 8px;
        text-align: left;
    }
    th {
        background-color: #f2f2f2;
    }
    .timestamp {
        font-size: 0.9em;
        color: #555555;
        margin-bottom: 20px;
    }
    hr {
        margin: 25px 0;
        border: none;
        border-top: 1px solid #cccccc;
    }
    pre {
        background: #f9f9f9;
        padding: 8px 10px;
        border-radius: 4px;
        white-space: pre-wrap;
    }
</style>
</head>
<body>
"""
        )
        f.write(f"<h1>{title}</h1>\n")
        f.write(
            f"<p class='timestamp'>Generated at: {datetime.now().isoformat()}</p>\n"
        )


def append_client_result(
    client_id: str,
    conclusion_text: str,
    summary_df: pd.DataFrame,
) -> None:
    with open(REPORT_PATH, "a", encoding="utf-8") as f:
        f.write(f"<h2>Client: {client_id}</h2>\n")
        f.write(f"<pre>{conclusion_text}</pre>\n")
        if not summary_df.empty:
            f.write(summary_df.to_html(index=False))
        f.write("<hr>\n")


def append_global_stats(global_df: pd.DataFrame) -> None:
    with open(REPORT_PATH, "a", encoding="utf-8") as f:
        f.write("<h2>Global Reliability Summary</h2>\n")
        if not global_df.empty:
            f.write(global_df.to_html(index=False))


def finish_report() -> None:
    with open(REPORT_PATH, "a", encoding="utf-8") as f:
        f.write("</body></html>\n")
