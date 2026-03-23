import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.training.finetune import main
from src.training.args import get_args_fine_tuning

if __name__ == "__main__":
    args = get_args_fine_tuning()
    main(args)