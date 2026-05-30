"""
internal/dashboard/analytics.py

Reads the diagnosis logs and generates analytics
for the Gradio dashboard tab.
"""

import logging
import pandas as pd
import plotly.express as px
from pathlib import Path
from config.config import settings

logger = logging.getLogger(__name__)


def load_logs() -> pd.DataFrame:
    """
    Load the diagnosis log CSV into a DataFrame.
    Returns an empty DataFrame if the log file doesn't exist yet.
    """
    path = Path(settings.diagnosis_log_path)

    if not path.exists():
        logger.warning("Diagnosis log file not found, returning empty DataFrame.")
        return pd.DataFrame()

    df = pd.read_csv(path)
    logger.info(f"Loaded {len(df)} diagnosis log entries.")
    return df


def diagnosis_distribution_chart():
    """
    Generate a bar chart showing the distribution of diagnosed disorders.
    Returns a Plotly figure or None if there is no data yet.
    """
    df = load_logs()

    if df.empty or "diagnosis" not in df.columns:
        logger.warning("No diagnosis data available for chart.")
        return None

    counts = df["diagnosis"].value_counts().reset_index()
    counts.columns = ["Diagnosis", "Count"]

    fig = px.bar(
        counts,
        x="Diagnosis",
        y="Count",
        title="Diagnosis Distribution",
        color="Diagnosis",
    )
    fig.update_layout(showlegend=False)
    return fig


def get_recent_logs(n: int = 100) -> pd.DataFrame:
    """Return the most recent n rows from the diagnosis log."""
    df = load_logs()

    if df.empty:
        return df

    return df.tail(n).reset_index(drop=True)