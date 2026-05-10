from pathlib import Path

import kagglehub

path = kagglehub.competition_download(
    "hms-harmful-brain-activity-classification",
    output_dir=str(Path("./data").resolve()),
)
print("Path to competition files:", path)
