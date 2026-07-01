# -*- coding: utf-8 -*-
"""Reusable package for the EEG + EMG robotic-arm BCI system."""

from .core import *
from .commands import *
from .gui import launch_gui
from .cli import add_common_robot_args, build_arg_parser, command_gui, main

