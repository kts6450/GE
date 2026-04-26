import subprocess
import sys
from pathlib import Path


SRC_DIR = Path(__file__).resolve().parent


def run_step(command: list[str]) -> None:
    print(f"\n>>> {' '.join(command)}")
    subprocess.run(command, cwd=SRC_DIR, check=True)


def main() -> None:
    python = sys.executable
    run_step([python, "collect_data.py"])
    run_step([python, "preprocess.py"])
    run_step([python, "train_model.py"])
    run_step([python, "predict.py", "--model", "dense"])


if __name__ == "__main__":
    main()
