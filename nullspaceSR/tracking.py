"""Optional Weights & Biases experiment tracking."""
import argparse


def add_wandb_arguments(parser):
    parser.add_argument("--wandb-enabled", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--wandb-project", default="NuSE-SR")
    parser.add_argument("--wandb-entity")
    parser.add_argument("--wandb-name")
    parser.add_argument("--wandb-mode", choices=("online", "offline", "disabled"), default="online")
    parser.add_argument("--wandb-tags", nargs="*", default=[])


def init_wandb(args, *, job_type):
    if not args.wandb_enabled or args.wandb_mode == "disabled":
        return None, None
    try:
        import wandb
    except ImportError as error:
        raise RuntimeError("W&B logging is enabled; install it with `python -m pip install -e '.[wandb]'`") from error
    run = wandb.init(project=args.wandb_project, entity=args.wandb_entity,
                     name=args.wandb_name, mode=args.wandb_mode,
                     tags=args.wandb_tags, job_type=job_type,
                     config=vars(args))
    run.define_metric("global_step")
    run.define_metric("train/*", step_metric="global_step")
    run.define_metric("validation/*", step_metric="epoch")
    run.define_metric("test/*")
    return wandb, run
