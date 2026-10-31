import json
import sys

from config import Config
from src.process.TargetGenerationProcess import plan_generation, run_target_generation


def main():
    if "--plan" in sys.argv[1:]:
        # Resolved configuration and what a run would do; nothing is created.
        print(json.dumps(plan_generation(), indent=2, sort_keys=True, default=str))
        return
    state = run_target_generation(Config)
    if state["status"] != "complete":
        sys.exit(1)


if __name__ == "__main__":
    main()
