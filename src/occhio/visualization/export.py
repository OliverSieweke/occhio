from datetime import datetime
from pathlib import Path
from typing import Any

import plotly.graph_objects as go

FIGURES_DIRECTORY = Path("../figures").resolve()


def export_figure(
    fig: go.Figure,
    labels: dict[str, Any],
    subdir: str | Path = "",
    dpi: int = 300,
    add_timestamp: bool = True,
    format: str = "png",
) -> Path:
    figures_path = Path(FIGURES_DIRECTORY) / subdir
    figures_path.mkdir(parents=True, exist_ok=True)

    filename_parts = []
    for key, value in sorted(labels.items()):
        # Sanitize value for filesystem
        safe_value = str(value).replace("/", "-").replace("\\", "-").replace(" ", "_")
        filename_parts.append(f"{key}={safe_value}")

    filename = "-".join(filename_parts)

    if add_timestamp:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        filename = f"{filename}_{timestamp}"

    filename = f"{filename}.{format}"
    filepath = figures_path / filename

    fig.write_image(filepath, format=format, scale=3)

    return filepath
