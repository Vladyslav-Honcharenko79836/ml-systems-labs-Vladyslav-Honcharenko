import torch

from common import ckpt_path
from models import TeacherCNN, make_student, make_teacher


def load_model(cfg, name, seed=42):
    if name == "baseline":
        m = make_teacher(cfg)
        m.load_state_dict(torch.load(ckpt_path(cfg, f"baseline_s{seed}"), map_location="cpu"))
    elif name.startswith(("global_", "layer_")):
        m = make_teacher(cfg)
        m.load_state_dict(torch.load(ckpt_path(cfg, f"{name}_s{seed}"), map_location="cpu"))
    elif name.startswith("structured_"):
        ck = torch.load(ckpt_path(cfg, f"{name}_s{seed}"), map_location="cpu")
        m = TeacherCNN(ck["widths"], cfg["baseline"]["hidden"], cfg["baseline"]["dropout"])
        m.load_state_dict(ck["state"])
    else:
        m = make_student(cfg)
        m.load_state_dict(torch.load(ckpt_path(cfg, f"{name}_s{seed}"), map_location="cpu"))
    return m.eval()


def names(cfg):
    p = cfg["prune"]
    sp = [s for s in p["sparsities"] if s > 0]
    return (["baseline"] + [f"global_{s}" for s in sp] + [f"layer_{s}" for s in sp]
            + [f"structured_{f}" for f in p["filter_fractions"]] + ["student_plain", "student_kd"])
