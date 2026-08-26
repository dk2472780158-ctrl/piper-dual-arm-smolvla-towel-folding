#!/usr/bin/env python

from dataclasses import dataclass, field

from lerobot.cameras import CameraConfig
from lerobot.cameras.realsense import RealSenseCameraConfig

from ..config import RobotConfig


@RobotConfig.register_subclass("piper_dual")
@dataclass
class PIPERDualConfig(RobotConfig):
    left_port: str = "can_left"
    right_port: str = "can_right"
    read_only: bool = False

    cameras: dict[str, CameraConfig] = field(
        default_factory=lambda: {
            "left": RealSenseCameraConfig(
                serial_number_or_name="261722072569",
                fps=30,
                width=640,
                height=480,
                use_depth=False,
                warmup_s=1,
            ),
            "right": RealSenseCameraConfig(
                serial_number_or_name="261622072187",
                fps=30,
                width=640,
                height=480,
                use_depth=False,
                warmup_s=1,
            ),
            "middle": RealSenseCameraConfig(
                serial_number_or_name="261822074601",
                fps=30,
                width=640,
                height=480,
                use_depth=False,
                warmup_s=1,
            ),
        }
    )
