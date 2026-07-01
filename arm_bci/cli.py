# -*- coding: utf-8 -*-
"""CLI argument parsing for the arm BCI system."""

from __future__ import annotations

from .commands import *
from .core import *
from .gui import launch_gui

def command_gui(args) -> None:
    code = launch_gui()
    if code:
        raise SystemExit(code)


def add_common_robot_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--target", default="target_1", help="Target area/object id sent to robot commands.")
    parser.add_argument("--robot-host", default="", help="Robot TCP host. If empty, use MockRobotArm.")
    parser.add_argument("--robot-port", type=int, default=9000, help="Robot TCP port.")
    parser.add_argument("--robot-timeout", type=float, default=1.0, help="Robot TCP timeout in seconds.")
    parser.add_argument("--mock-delay", type=float, default=0.15, help="Mock robot action delay.")
    parser.add_argument("--refractory-seconds", type=float, default=0.25, help="Minimum seconds between accepted movement pulses.")
    parser.add_argument("--command-duration", type=float, default=0.55, help="Seconds that each accepted movement command remains active.")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Hybrid masseter-plus-MI rest-gated 6-movement robotic arm control system")
    sub = parser.add_subparsers(dest="command")

    p = sub.add_parser("gui", help="Open the PyQt multi-page GUI")
    p.set_defaults(func=command_gui)

    p = sub.add_parser("train", help="Train calibration model from npz data")
    p.add_argument("--data", required=True, help="Calibration npz with eeg, y, optional aux, optional fs")
    p.add_argument("--model", default="models/arm_bci_model.joblib", help="Output model path")
    p.add_argument("--fs", type=int, default=250, help="Sampling rate fallback if npz has no fs")
    p.add_argument("--classifier", choices=["logreg", "lda"], default="logreg")
    p.add_argument("--eeg-weight", type=float, default=0.55)
    p.add_argument("--command-confidence", type=float, default=0.62)
    p.add_argument("--stable-windows", type=int, default=2)
    p.add_argument("--refractory-seconds", type=float, default=0.25)
    p.add_argument("--command-duration", type=float, default=0.55)
    p.add_argument("--no-csp", action="store_true", help="Disable optional MNE CSP features")
    p.set_defaults(func=command_train)

    p = sub.add_parser("evaluate", help="Evaluate a saved model on npz data")
    p.add_argument("--data", required=True)
    p.add_argument("--model", required=True)
    p.set_defaults(func=command_evaluate)

    p = sub.add_parser("make-demo-data", help="Create synthetic calibration data for a smoke test")
    p.add_argument("--out", default="data/processed/calibration_demo.npz")
    p.add_argument("--fs", type=int, default=250)
    p.add_argument("--trials-per-class", type=int, default=16)
    p.set_defaults(func=command_make_demo_data)

    p = sub.add_parser("simulate", help="Run a full mock online loop")
    p.add_argument("--model", default="models/arm_bci_demo.joblib")
    p.add_argument("--fs", type=int, default=250)
    p.add_argument("--trials-per-class", type=int, default=16)
    p.add_argument("--cycles", type=int, default=2)
    p.add_argument("--realtime", action="store_true")
    add_common_robot_args(p)
    p.set_defaults(func=command_simulate)

    p = sub.add_parser("online", help="Run online decoding from replay npz or hardware source")
    p.add_argument("--model", required=True)
    p.add_argument("--replay", default="", help="Replay npz with continuous/trial eeg and optional aux")
    p.add_argument("--rda", action="store_true", help="Read continuous 64-channel BrainVision RDA stream")
    p.add_argument("--rda-host", default="127.0.0.1")
    p.add_argument("--rda-port", type=int, default=51244)
    p.add_argument("--channels", type=int, default=64)
    p.add_argument("--aux-channels", type=int, default=3)
    p.add_argument("--realtime", action="store_true")
    add_common_robot_args(p)
    p.set_defaults(func=command_online)

    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    if argv is None:
        argv = sys.argv[1:]
    if len(argv) == 0:
        return launch_gui()
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    try:
        args.func(args)
        return 0
    except KeyboardInterrupt:
        print("Interrupted")
        return 130
    except Exception as exc:
        print(f"ERROR: {exc}")
        return 1


