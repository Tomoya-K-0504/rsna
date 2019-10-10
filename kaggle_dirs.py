from pathlib import Path


dirs = ["input/processed", "output", "model", "notebook", "src", "log", "report"]
for dir in dirs:
    Path(dir).mkdir(parents=True, exist_ok=True)
