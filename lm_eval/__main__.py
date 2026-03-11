from lm_eval._cli import HarnessCLI
from lm_eval.utils import setup_logging


def cli_evaluate() -> None:
    """Main CLI entry point."""
    setup_logging()
    parser = HarnessCLI()
    args = parser.parse_args()
    try:
        parser.execute(args)
    finally:
        # Clean up distributed process group to avoid resource leak warning
        import contextlib

        with contextlib.suppress(Exception):
            import torch.distributed as dist

            if dist.is_initialized():
                dist.destroy_process_group()


if __name__ == "__main__":
    cli_evaluate()
