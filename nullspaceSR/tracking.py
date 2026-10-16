"""Optional Weights & Biases experiment tracking."""
import argparse
import torch


def wandb_rgb_image(wandb, image, *, caption):
    """Serialize a CHW float image as explicit RGB uint8 HWC for W&B."""
    image = image.detach().to(device="cpu", dtype=torch.float32)
    if image.ndim != 3 or image.shape[0] != 3:
        raise ValueError(f"W&B preview expects a 3-channel CHW image, got {tuple(image.shape)}")
    pixels = image.clamp(0, 1).mul(255).round().to(dtype=torch.uint8)
    pixels = pixels.permute(1, 2, 0).contiguous().numpy()
    return wandb.Image(pixels, mode="RGB", caption=caption, normalize=False)


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
