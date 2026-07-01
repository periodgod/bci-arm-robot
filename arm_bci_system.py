# -*- coding: utf-8 -*-
"""Compatibility launcher for the modular arm_bci package."""

from __future__ import annotations

from arm_bci.core import *
from arm_bci.commands import *
from arm_bci.gui import launch_gui
from arm_bci.cli import add_common_robot_args, build_arg_parser, command_gui, main


if __name__ == "__main__":
    raise SystemExit(main())

