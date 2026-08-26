import os
import time
from dataclasses import dataclass

from piper_sdk import C_PiperInterface_V2


@dataclass
class PiperMotorsBusConfig:
    can_name: str
    motors: dict[str, tuple[int, str]]


class PiperMotorsBus:
    """Piper SDK secondary wrapper."""

    def __init__(self, config: PiperMotorsBusConfig):
        # Enable Piper SDK 0.6.2 as the final hardware-limit layer for
        # follower feedback and control commands.
        self.piper = C_PiperInterface_V2(
            config.can_name,
            start_sdk_joint_limit=True,
            start_sdk_gripper_limit=True,
        )
        self.piper.ConnectPort()
        self._is_connected = True
        self.motors = config.motors
        self.init_joint_position = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        self.safe_disable_position = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        self.pose_factor = 1000
        self.joint_factor = 57324.840764

        # Disabled by default so data collection and replay remain unchanged.
        # Set PIPER_ACTION_FILTER_ALPHA below 1.0 only for policy inference.
        self.action_filter_alpha = float(os.environ.get("PIPER_ACTION_FILTER_ALPHA", "1.0"))
        if not 0.0 < self.action_filter_alpha <= 1.0:
            raise ValueError("PIPER_ACTION_FILTER_ALPHA must be in the range (0, 1]")
        self._filtered_joint_target: list[float] | None = None
        if self.action_filter_alpha < 1.0:
            print(
                f"Piper joint filter enabled on {config.can_name}: "
                f"alpha={self.action_filter_alpha:.3f}"
            )

    @property
    def is_connected(self) -> bool:
        return self._is_connected

    @property
    def motor_names(self) -> list[str]:
        return list(self.motors.keys())

    @property
    def motor_models(self) -> list[str]:
        return [model for _, model in self.motors.values()]

    @property
    def motor_indices(self) -> list[int]:
        return [idx for idx, _ in self.motors.values()]

    def connect(self, enable: bool) -> bool:
        """Enable Piper and check the enable state for up to five seconds."""
        enable_flag = False
        loop_flag = False
        timeout = 5
        start_time = time.time()
        while not loop_flag:
            elapsed_time = time.time() - start_time
            print("--------------------")
            enable_list = []
            enable_list.append(self.piper.GetArmLowSpdInfoMsgs().motor_1.foc_status.driver_enable_status)
            enable_list.append(self.piper.GetArmLowSpdInfoMsgs().motor_2.foc_status.driver_enable_status)
            enable_list.append(self.piper.GetArmLowSpdInfoMsgs().motor_3.foc_status.driver_enable_status)
            enable_list.append(self.piper.GetArmLowSpdInfoMsgs().motor_4.foc_status.driver_enable_status)
            enable_list.append(self.piper.GetArmLowSpdInfoMsgs().motor_5.foc_status.driver_enable_status)
            enable_list.append(self.piper.GetArmLowSpdInfoMsgs().motor_6.foc_status.driver_enable_status)
            if enable:
                enable_flag = all(enable_list)
                while not self.piper.EnablePiper():
                    print("piper initing")
                    time.sleep(0.1)
                self.piper.GripperCtrl(0, 1000, 0x01, 0)
            else:
                enable_flag = any(enable_list)
                self.piper.DisableArm(7)
                self.piper.GripperCtrl(0, 1000, 0x02, 0)
            print(f"使能状态: {enable_flag}")
            print("--------------------")
            if enable_flag == enable:
                loop_flag = True
                enable_flag = True
            else:
                loop_flag = False
                enable_flag = False
            if elapsed_time > timeout:
                print("超时....")
                enable_flag = False
                loop_flag = True
                break
            time.sleep(0.5)
        print(f"Returning response: {enable_flag}")
        self._is_connected = enable_flag
        return enable_flag

    def set_calibration(self):
        return

    def revert_calibration(self):
        return

    def apply_calibration(self):
        """Move Piper to the initial position."""
        self.write(target_joint=self.init_joint_position)

    def _filter_target(self, target_joint: list) -> list[float]:
        """Low-pass only the six arm joints; never delay the gripper."""
        if len(target_joint) != 7:
            raise ValueError(f"Expected 7 Piper targets, got {len(target_joint)}")

        target = [float(value) for value in target_joint]
        new_joints = target[:6]
        previous = self._filtered_joint_target

        if previous is None or self.action_filter_alpha >= 1.0:
            filtered_joints = new_joints
        else:
            alpha = self.action_filter_alpha
            filtered_joints = [
                alpha * new + (1.0 - alpha) * old
                for new, old in zip(new_joints, previous, strict=True)
            ]

        self._filtered_joint_target = filtered_joints.copy()
        return filtered_joints + [target[6]]

    def write(self, target_joint: list):
        """Send a seven-dimensional position target in radians/meters."""
        target_joint = self._filter_target(target_joint)

        joint_0 = round(target_joint[0] * self.joint_factor)
        joint_1 = round(target_joint[1] * self.joint_factor)
        joint_2 = round(target_joint[2] * self.joint_factor)
        joint_3 = round(target_joint[3] * self.joint_factor)
        joint_4 = round(target_joint[4] * self.joint_factor)
        joint_5 = round(target_joint[5] * self.joint_factor)
        gripper_range = round(target_joint[6] * 1000 * 1000)

        self.piper.MotionCtrl_2(0x01, 0x01, 50, 0x00)
        self.piper.JointCtrl(joint_0, joint_1, joint_2, joint_3, joint_4, joint_5)
        self.piper.GripperCtrl(abs(gripper_range), 1000, 0x01, 0)

    def read(self) -> dict:
        """Read joint/gripper positions and efforts."""
        joint_msg = self.piper.GetArmJointMsgs()
        joint_state = joint_msg.joint_state
        gripper_msg = self.piper.GetArmGripperMsgs()
        gripper_state = gripper_msg.gripper_state
        high_spd_msg = self.piper.GetArmHighSpdInfoMsgs()

        return {
            "joint_1_pos": joint_state.joint_1 / self.joint_factor,
            "joint_2_pos": joint_state.joint_2 / self.joint_factor,
            "joint_3_pos": joint_state.joint_3 / self.joint_factor,
            "joint_4_pos": joint_state.joint_4 / self.joint_factor,
            "joint_5_pos": joint_state.joint_5 / self.joint_factor,
            "joint_6_pos": joint_state.joint_6 / self.joint_factor,
            "gripper_pos": gripper_state.grippers_angle / 1000000.0,
            "joint_1_effort": high_spd_msg.motor_1.effort / 1000.0,
            "joint_2_effort": high_spd_msg.motor_2.effort / 1000.0,
            "joint_3_effort": high_spd_msg.motor_3.effort / 1000.0,
            "joint_4_effort": high_spd_msg.motor_4.effort / 1000.0,
            "joint_5_effort": high_spd_msg.motor_5.effort / 1000.0,
            "joint_6_effort": high_spd_msg.motor_6.effort / 1000.0,
            "gripper_effort": gripper_state.grippers_effort / 1000.0,
        }

    def safe_disconnect(self):
        """Move to the configured safe disconnect position."""
        self.write(target_joint=self.safe_disable_position)
