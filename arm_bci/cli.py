# -*- coding: utf-8 -*-
"""Entry point for the single EEG8 control-console workflow."""

from __future__ import annotations

import argparse
import sys
from typing import Optional, Sequence

from .gui import launch_gui


def command_gui(_args: argparse.Namespace) -> None:
    code = launch_gui()
    if code:
        raise SystemExit(code)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="EEG8 多模态机械臂 BCI 控制台",
    )
    sub = parser.add_subparsers(dest="command")
    gui = sub.add_parser("gui", help="启动 EEG8 图形控制台")
    gui.set_defaults(func=command_gui)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args_in = list(sys.argv[1:] if argv is None else argv)
    if not args_in:
        return launch_gui()
    parser = build_arg_parser()
    args = parser.parse_args(args_in)
    if not hasattr(args, "func"):
        parser.print_help()
        return 2
    try:
        args.func(args)
        return 0
    except KeyboardInterrupt:
        return 130
    except Exception as exc:
        print(f"ERROR: {exc}")
        return 1
