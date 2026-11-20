# -*- coding: utf-8 -*-
"""Compatibility launcher for the modular arm_bci package."""

from __future__ import annotations

from arm_bci.gui import launch_gui
from arm_bci.cli import build_arg_parser, command_gui, main


if __name__ == "__main__":
    raise SystemExit(main()) 

9