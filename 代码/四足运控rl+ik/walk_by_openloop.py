#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Leika simulation (play/controller/gait/kinematics) 同构的真机开环脚本。
只要确保一个前提：
- URDF 中的关节定义顺序与你实机中真实的电机总线 ID 排列对齐。
"""

import argparse
import math
import os
import sys
import time
from enum import Enum
from typing import Dict, List, TypedDict

import numpy as np

# ==========================================
# 1. 动态调整 Python 搜索路径
# ==========================================
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
    
from runtime.position_hwi import HWI

# ==========================================
# 2. 传统步态控制参数配置
# ==========================================
class GaitType(Enum):
    TROT = 0   # 小跑机制（两两对角腿同步跨步）
    CRAWL = 1  # 爬行机制（四条腿按 90° 相位差依次独立跨步）


# 四条腿在整个控制周期（0.0 ~ 1.0）中的起步相位偏置
default_offset = {
    GaitType.TROT: [0, 0.5, 0.5, 0],              # 对角腿相位互补（相差 180 度）
    GaitType.CRAWL: [0, 1 / 4, 2 / 4, 3 / 4],     # 四条腿相位均匀分布，依次抬起
}

# 支撑相占比 (Stance Fraction): 单脚踩地支撑的时间占总步态周期的比例
default_stand_frac = {
    GaitType.TROT: 3 / 4,                         # 75% 时间踩地支撑，25% 时间空中摆动
    GaitType.CRAWL: 3 / 4,
}


class KinConfig:
    """ 
    机器狗几何连杆尺寸限制及初始位姿参数 (单位: 米 / 弧度)
    """
    coxa = 60.5 / 100.0         # 侧向髋关节连杆长度
    coxa_offset = 10.0 / 100.0  # 髋关节偏置距离
    femur = 111.2 / 100.0       # 大腿骨架连杆长度
    tibia = 118.5 / 100.0       # 小腿连杆长度
    L = 207.5 / 100.0           # 躯干轴距（前后长度）
    W = 78.0 / 100.0            # 躯干轮距（左右宽度）

    # 4 条腿在躯干中心坐标系下的物理挂载基座位置
    mount_offsets = np.array([[L / 2, 0, W / 2], [L / 2, 0, -W / 2], [-L / 2, 0, W / 2], [-L / 2, 0, -W / 2]])

    # 静态参考下的标准初始支撑脚位置
    default_feet_positions = np.array(
        [
            [mount_offsets[0][0], 0, mount_offsets[0][2] + coxa],
            [mount_offsets[1][0], 0, mount_offsets[1][2] - coxa],
            [mount_offsets[2][0], 0, mount_offsets[2][2] + coxa],
            [mount_offsets[3][0], 0, mount_offsets[3][2] - coxa],
        ]
    )

    # 运动解算的安全幅值限制范围
    max_roll = 15 * np.pi / 2
    max_pitch = 15 * np.pi / 2
    max_body_shift_x = W / 3
    max_body_shift_z = W / 3
    max_leg_reach = femur + tibia - coxa_offset
    min_body_height = max_leg_reach * 0.45
    max_body_height = max_leg_reach * 0.9
    body_height_range = max_body_height - min_body_height
    max_step_length = max_leg_reach * 0.8
    max_step_height = max_leg_reach / 2

    # 默认开机静态高度与跨步参数
    default_step_depth = 0.002
    default_body_height = min_body_height + body_height_range / 2
    default_step_height = default_body_height / 2


# 机身 6 自由度与足端状态强类型定义
class BodyState(TypedDict):
    omega: float; phi: float; psi: float  # Roll, Pitch, Yaw
    xm: float; ym: float; zm: float        # X, Y, Z 轴机身平移量
    px: float; py: float; pz: float        # 身体旋转几何轴心偏置
    feet: np.ndarray                       # 四个足端的当前实时控制目标坐标
    default_feet: np.ndarray               # 静态标准的足端坐标基准


class GaitState(TypedDict):
    step_height: float; step_x: float; step_z: float; step_angle: float; step_depth: float
    stand_frac: float
    offset: List[float]
    gait_type: GaitType


# ==========================================
# 3. 足端空间运动轨迹生成算法
# ==========================================
# 贝塞尔控制点的平移拉伸向量乘数
length_multipliers = np.array([-1.4, -1.0, -1.5, -1.5, -1.5, 0.0, 0.0, 0.0, 1.5, 1.5, 1.4, 1.0])
# 贝塞尔控制点的垂直抛物线高度比率分配系数
height_profile = np.array([0.0, 0.0, 0.9, 0.9, 0.9, 0.9, 0.9, 1.1, 1.1, 1.1, 0.0, 0.0])


def sine_curve(length, angle, depth, phase):
    """ 
    支撑相蹬地轨迹 (Stance Trajectory):
    当脚踩在地上时，利用半周期余弦波动曲线模拟向后蹬地并带有轻微向下压实（depth）的力学过程。
    """
    x_polar = np.cos(angle)
    z_polar = np.sin(angle)
    step = length * (1 - 2 * phase)
    x = step * x_polar
    z = step * z_polar
    y = -depth * np.cos((np.pi * (x + z)) / (2 * length)) if length != 0 else 0
    return np.array([x, y, z])


def yaw_arc(default_foot, current_foot):
    """ 原地打转 (Yaw 转向) 时的切线轨迹圆弧几何偏角计算 """
    foot_mag = np.sqrt(default_foot[0] ** 2 + default_foot[2] ** 2)
    foot_dir = np.arctan2(default_foot[2], default_foot[0])
    offset_x = current_foot[0] - default_foot[0]
    offset_z = current_foot[2] - default_foot[2]
    offset_mag = np.sqrt(offset_x**2 + offset_z**2)
    offset_mod = np.arctan2(offset_mag, foot_mag)
    return np.pi / 2.0 + foot_dir + offset_mod


def get_control_points(length, angle, height):
    """ 生成 12 阶贝塞尔曲线所依托的空间控制点坐标矩阵 """
    x_polar = np.cos(angle)
    z_polar = np.sin(angle)
    x = length * length_multipliers * x_polar
    z = length * length_multipliers * z_polar
    y = height * height_profile
    return np.stack([x, y, z], axis=1)


def bezier_curve(length, angle, height, phase):
    """
    摆动相迈腿轨迹 (Swing Trajectory):
    根据多阶伯恩斯坦多项式对空间控制点插值求和，计算出在空中划过的极度平滑的悬空抬腿轨迹。
    """
    ctrl = get_control_points(length, angle, height)
    n = len(ctrl) - 1
    # 伯恩斯坦多项式组合数计算系数
    coeffs = np.array([math.comb(n, i) * (phase**i) * ((1 - phase) ** (n - i)) for i in range(n + 1)])
    return np.sum(ctrl * coeffs[:, None], axis=0)


# ==========================================
# 4. 步态时序状态机 (Gait Controller)
# ==========================================
class GaitController:
    def __init__(self, default_position: np.ndarray):
        self.default_position = default_position.copy()
        self.phase = 0.0 # 全局归一化时间相位时钟 (0.0 ~ 1.0 循环)

    def step(self, gait: GaitState, body: BodyState, dt: float):
        """ 步态时钟演进：计算四条腿下一时刻应该踏在三维空间中的哪个坐标点 """
        step_x, step_z, angle = gait["step_x"], gait["step_z"], gait["step_angle"]
        
        # 没有任何运动指令时，足端迅速但平滑地收回至出厂默认位置
        if not any((step_x, step_z, angle)):
            body["feet"] = body["feet"] + (self.default_position - body["feet"]) * dt * 10
            self.phase = 0.0
            return

        period = 2.0  # 走完一轮完整跨步周期需要 2.0 秒
        self.phase = (self.phase + dt / period) % 1 # 时间向前推进

        stand_fraction = gait["stand_frac"]
        depth = gait["step_depth"]
        height = gait["step_height"]
        offsets = gait["offset"]

        length = np.hypot(step_x, step_z)
        if step_x < 0:
            length = -length
        turn_amplitude = np.arctan2(step_z, length if length != 0 else 1e-8)

        new_feet = self.default_position.copy()
        
        # 分别解算 4 条腿各自的运动身姿
        for i, (default_foot, current_foot) in enumerate(zip(self.default_position, body["feet"])):
            # 加上各腿之间固有的相位偏移量，获取单腿当前时钟
            phase = (self.phase + offsets[i]) % 1
            
            if phase < stand_fraction:
                # 状态 1：当前腿局部时间属于支撑状态 -> 采用正弦蹬地
                ph_norm, curve_fn, amp = phase / stand_fraction, sine_curve, -depth
            else:
                # 状态 2：当前腿局部时间属于空中摆动 -> 采用贝塞尔悬空跨步
                ph_norm, curve_fn, amp = (phase - stand_fraction) / (1 - stand_fraction), bezier_curve, height

            # 分别算出平移步长和原地打转的足端修正量
            delta_pos = curve_fn(length / 2, turn_amplitude, amp, ph_norm)
            delta_rot = curve_fn(angle * 2, yaw_arc(default_foot, current_foot), amp, ph_norm)

            # 叠加融合，得到该腿最终的新空间坐标
            new_feet[i][0] = default_foot[0] + delta_pos[0] + delta_rot[0] * 0.2
            new_feet[i][2] = default_foot[2] + delta_pos[2] + delta_rot[2] * 0.2
            if length or angle:
                new_feet[i][1] = default_foot[1] + delta_pos[1] + delta_rot[1] * 0.2

        body["feet"] = new_feet


# ==========================================
# 5. 全身运动几何几何逆解 (Kinematics)
# ==========================================
class Kinematics:
    def __init__(self):
        self.coxa = KinConfig.coxa
        self.coxa_offset = KinConfig.coxa_offset
        self.femur = KinConfig.femur
        self.tibia = KinConfig.tibia
        self.mount_offsets = KinConfig.mount_offsets.copy()
        # 旋转基准变换矩阵：用于对齐机械结构与数学坐标系
        self.inv_mount_rot = np.array([[0, 0, -1], [0, 1, 0], [1, 0, 0]])

    def inverse_kinematics(self, body_state: BodyState):
        """ 全身运动学逆解：将足端世界坐标系下的绝对路径，反向映射解算出 12 个电机的目标控制角度 """
        roll, pitch, yaw = np.deg2rad(body_state["omega"]), np.deg2rad(body_state["phi"]), np.deg2rad(body_state["psi"])
        xm, ym, zm = body_state["xm"], body_state["ym"], body_state["zm"]

        # 构建 3 轴姿态矩阵，计算机身扭动对腿部基座造成的位移变化
        rot = self._rotation_matrix(roll, pitch, yaw)
        inv_rot = rot.T
        inv_tr = -inv_rot @ np.array([xm, ym, zm])

        angles = []
        # 循环解算四条腿
        for idx, foot_world in enumerate(body_state["feet"]):
            # 坐标齐次坐标变换：世界位置 -> 机身相对位置 -> 单腿局部基座位置
            foot_body = inv_rot @ foot_world + inv_tr
            foot_local = self.inv_mount_rot @ (foot_body - self.mount_offsets[idx])
            
            # 左右腿由于是对称镜像组装，X 轴输入坐标需要条件取反
            x_local = -foot_local[0] if idx % 2 else foot_local[0]
            
            # 几何三角解析法解算单腿的 3 个关节角
            angles.extend(self._leg_ik(x_local, foot_local[1], foot_local[2]))
        return np.array(angles, dtype=float)

    def _leg_ik(self, x, y, z):
        """ 单腿 3 自由度连杆解析几何公式（利用解析法和余弦定理） """
        f = np.sqrt(max(0.0, x * x + y * y - self.coxa * self.coxa))
        g = f - self.coxa_offset
        h = np.sqrt(g * g + z * z)

        # 侧向滚转髋关节角
        theta1 = -np.arctan2(y, x) - np.arctan2(f, -self.coxa)
        # 余弦定理求膝关节（小腿连杆）偏角
        d = (h * h - self.femur * self.femur - self.tibia * self.tibia) / (2 * self.femur * self.tibia)
        theta3 = np.arccos(max(-1.0, min(1.0, d)))
        # 大腿连杆偏角
        theta2 = np.arctan2(z, g) - np.arctan2(self.tibia * np.sin(theta3), self.femur + self.tibia * np.cos(theta3))
        return theta1, theta2, theta3

    def _rotation_matrix(self, roll, pitch, yaw):
        """ 标准的欧拉角转旋转矩阵计算 """
        cr, sr = np.cos(roll), np.sin(roll)
        cp, sp = np.cos(pitch), np.sin(pitch)
        cy, sy = np.cos(yaw), np.sin(yaw)
        return np.array(
            [
                [cp * cy, -cp * sy, sp],
                [sr * sp * cy + sy * cr, -sr * sp * sy + cr * cy, -sr * cp],
                [sr * sy - sp * cr * cy, sr * cy + sp * sy * cr, cr * cp],
            ]
        )


# ==========================================
# 6. 拓扑与指令映射对齐接口
# ==========================================
# 仿真引擎中标准导出的 12 关节一维数组物理拓扑序列
URDF_JOINT_ORDER = [
    "motor_front_left_shoulder", "motor_front_left_leg", "foot_motor_front_left",
    "motor_front_right_shoulder", "motor_front_right_leg", "foot_motor_front_right",
    "motor_rear_left_shoulder", "motor_rear_left_leg", "foot_motor_rear_left",
    "motor_rear_right_shoulder", "motor_rear_right_leg", "foot_motor_rear_right",
]

# 将仿真命名转换为 HWI 绑定的实机统一关节命名空间
URDF_TO_LOCAL_JOINT = {
    "motor_front_left_shoulder": "left_front_hip_joint",
    "motor_front_left_leg": "left_front_knee_joint",
    "foot_motor_front_left": "left_front_ankle_joint",
    "motor_front_right_shoulder": "right_front_hip_joint",
    "motor_front_right_leg": "right_front_knee_joint",
    "foot_motor_front_right": "right_front_ankle_joint",
    "motor_rear_left_shoulder": "left_back_hip_joint",
    "motor_rear_left_leg": "left_back_knee_joint",
    "foot_motor_rear_left": "left_back_ankle_joint",
    "motor_rear_right_shoulder": "right_back_hip_joint",
    "motor_rear_right_leg": "right_back_knee_joint",
    "foot_motor_rear_right": "right_back_ankle_joint",
}

JOINT_ORDER = [URDF_TO_LOCAL_JOINT[name] for name in URDF_JOINT_ORDER]


def joints_to_local_cmd(joints_12: np.ndarray) -> Dict[str, float]:
    """ 运动学解算角度数组映射组合为带命名的指令字典 """
    return {name: float(joints_12[i]) for i, name in enumerate(JOINT_ORDER)}


def mech_to_servo(hwi, joint_name: str, mech_angle: float) -> float:
    """
    终极标定核心转换：
    将数学层完全理想对称的运动学角度（mech_angle），乘上安装方向取反系数（sign），
    叠加开机默认姿态基准以及物理装配调零偏移量，计算出真正写入飞特总线物理总线上的执行弧度。
    """
    real_pose = hwi.real_pose[joint_name]
    sign = {
        "right_front_hip_joint": -1.0,
        "right_front_knee_joint": -1.0,  
        "right_front_ankle_joint": -1.0,

        "left_front_hip_joint": 1.0,
        "left_front_knee_joint": 1.0,
        "left_front_ankle_joint": 1.0,

        "left_back_hip_joint": -1.0,
        "left_back_knee_joint": 1.0,
        "left_back_ankle_joint": 1.0,


        "right_back_hip_joint": 1.0,
        "right_back_knee_joint": -1.0,
        "right_back_ankle_joint": -1.0,
        }

    sign = sign[joint_name]

    return (mech_angle - real_pose) * sign + hwi.init_pos[joint_name] + hwi.joints_offsets[joint_name]


def send_local_cmd_ordered(hwi, local_cmd: Dict[str, float]):
    """ 按照底层总线的物理拓扑 ID 顺序打包统一广播给所有关节 """
    ordered_ids = [hwi.joints[name] for name in JOINT_ORDER]
    ordered_positions = [mech_to_servo(hwi, name, local_cmd[name]) for name in JOINT_ORDER]
    hwi.io.write_goal_position(ordered_ids, ordered_positions)


# ==========================================
# 7. PyBullet 中央调试控制滑块面板创建pip install pybullet -i https://pypi.tuna.tsinghua.edu.cn/simple --trusted-host pypi.tuna.tsinghua.edu.cn
# ==========================================
def setup_gui_sliders():
    """ 调阅 PyBullet 窗口渲染组件，在终端外侧生成调试用鼠标控制滑块 """
    import pybullet as p

    # 若尚未连接任何独立的物理世界，静默挂载一个纯空的 GUI 窗口来捕获滑块及按键
    if p.getConnectionInfo()["isConnected"] == 0:
        p.connect(p.GUI)

    sliders = {
        "x": p.addUserDebugParameter("x", -1, 1, 0),                       # 控制狗整体质心前后平移
        "y": p.addUserDebugParameter("y", 0, 1, 0.5),                      # 控制狗高度
        "z": p.addUserDebugParameter("z", -1, 1, 0),                       # 控制狗质心左右平移
        "yaw": p.addUserDebugParameter("yaw", -1, 1, 0),                   # 改变机身绝对 Yaw 偏航
        "pitch": p.addUserDebugParameter("pitch", -1, 1, 0),               # 改变机身绝对 Pitch 俯仰
        "roll": p.addUserDebugParameter("roll", -1, 1, 0),                 # 改变机身绝对 Roll 翻滚
        "pivot_x": p.addUserDebugParameter("pivot x", -1, 1, 0),
        "pivot_y": p.addUserDebugParameter("pivot y", -1, 1, 0),
        "pivot_z": p.addUserDebugParameter("pivot z", -1, 1, 0),
        "step_x": p.addUserDebugParameter("Step x", -1, 1, 0),             # 行走前行步长/速度控制
        "step_z": p.addUserDebugParameter("Step z", -1, 1, 0),             # 横向平移步长速度控制
        "angle": p.addUserDebugParameter("Angle", -1, 1, 0),               # 行进间边走边拐弯的转弯半径控制
        "step_height": p.addUserDebugParameter("Step height", 0, 1, 0.5),   # 抬腿跨步高度
        "step_depth": p.addUserDebugParameter("Step depth", 0, 0.01, 0.002),# 蹬地深陷比率
        "stand_frac": p.addUserDebugParameter("Stand frac", 0, 1, 0.75),   # 站立支撑占比
    }
    return p, sliders


def update_state_from_gui(p, sliders, body_state: BodyState, gait_state: GaitState):
    """ 捕获用户在窗口滑块上的拖拽数值，映射重算物理结构参数 """
    gait_state["step_x"] = p.readUserDebugParameter(sliders["step_x"]) * KinConfig.max_step_length
    gait_state["step_z"] = p.readUserDebugParameter(sliders["step_z"]) * KinConfig.max_step_length
    gait_state["step_angle"] = p.readUserDebugParameter(sliders["angle"])
    gait_state["step_height"] = p.readUserDebugParameter(sliders["step_height"]) * KinConfig.max_step_height
    gait_state["step_depth"] = p.readUserDebugParameter(sliders["step_depth"])
    gait_state["stand_frac"] = p.readUserDebugParameter(sliders["stand_frac"])
    gait_state["offset"] = default_offset[GaitType.TROT]
    gait_state["gait_type"] = GaitType.TROT

    body_state["xm"] = p.readUserDebugParameter(sliders["x"]) * KinConfig.max_body_shift_x
    body_state["ym"] = p.readUserDebugParameter(sliders["y"]) * KinConfig.body_height_range + KinConfig.min_body_height
    body_state["zm"] = p.readUserDebugParameter(sliders["z"]) * KinConfig.max_body_shift_z
    body_state["omega"] = p.readUserDebugParameter(sliders["roll"]) * KinConfig.max_roll
    body_state["phi"] = p.readUserDebugParameter(sliders["pitch"]) * KinConfig.max_pitch
    body_state["psi"] = p.readUserDebugParameter(sliders["yaw"]) * KinConfig.max_pitch
    body_state["px"] = p.readUserDebugParameter(sliders["pivot_x"]) * KinConfig.max_body_shift_x
    body_state["py"] = p.readUserDebugParameter(sliders["pivot_y"]) * KinConfig.max_body_shift_x
    body_state["pz"] = p.readUserDebugParameter(sliders["pivot_z"]) * KinConfig.max_body_shift_z


# ==========================================
# 8. 实机遥控主控制循环 (50Hz Teleop Loop)
# ==========================================
def main():
    parser = argparse.ArgumentParser(description="Leika 传统的开环模型位置解算真机控制入口")
    parser.add_argument("--usb-port", default="/dev/ttyACM0")
    parser.add_argument("--duration", type=float, default=0.0, help="运行持续秒数，0 代表长久无限循环")
    parser.add_argument("--dt", type=float, default=0.02) # 严格的主控控制频率周期：0.02秒 (等价于 50Hz 控制环)
    args = parser.parse_args()

    gait_type = GaitType.TROT
    standby = KinConfig.default_feet_positions.copy()

    # 数据包状态结构体初始化
    body_state: BodyState = {
        "omega": 0.0, "phi": 0.0, "psi": 0.0,
        "xm": 0.0, "ym": KinConfig.default_body_height, "zm": 0.0,
        "px": 0.0, "py": 0.0, "pz": 0.0,
        "feet": standby.copy(),
        "default_feet": standby.copy(),
    }

    gait_state: GaitState = {
        "step_height": KinConfig.default_step_height,
        "step_x": 0.0, "step_z": 0.0, "step_angle": 0.0,
        "step_depth": KinConfig.default_step_depth,
        "stand_frac": default_stand_frac[gait_type],
        "offset": default_offset[gait_type],
        "gait_type": gait_type,
    }

    # 运动解算三大件实例化
    gait = GaitController(standby)   # 1) 步态发生器
    ik = Kinematics()                 # 2) 逆运动学变换核心
    hwi = HWI(usb_port=args.usb_port) # 3) 底层串口控制器

    # 安全慢通电让狗平稳支撑站立
    hwi.turn_on()
    pb, sliders = setup_gui_sliders() # 调出调试面板

    # 首次开机无速度零点标定
    joints = ik.inverse_kinematics(body_state)
    target_cmd = joints_to_local_cmd(joints)

    start_t = time.time()
    next_t = start_t

    try:
        while True:
            now = time.time()
            if args.duration > 0.0 and now - start_t >= args.duration:
                break

            # 8.1 抓取 GUI 界面滑块最新状态
            update_state_from_gui(pb, sliders, body_state, gait_state)

            # 支持在控制面板上聚焦按键盘 Q 键或 Esc 键安全退出
            keys = pb.getKeyboardEvents()
            if keys.get(ord("q")) or keys.get(27):
                print("[INFO] GUI quit key pressed")
                break

            # 8.2 根据滑块命令计算下一时刻足端应该挪移至哪个理想物理坐标
            gait.step(gait_state, body_state, args.dt)

            # 8.3 【核心动作】：调用全身连杆运动解析公式，反向解算出 12 个关节此时此刻的目标偏角
            joints = ik.inverse_kinematics(body_state)
            
            # 8.4 对齐关节命名空间
            local_cmd = joints_to_local_cmd(joints)
            
            # 8.5 计算安装符号偏置后，将一整包同步位置数据通过 1Mbps 高速串口塞入物理飞特舵机
            send_local_cmd_ordered(hwi, local_cmd)

            # 8.6 工业级严格主控循环时间对齐控制器
            next_t += args.dt
            sleep_t = next_t - time.time()
            if sleep_t > 0:
                time.sleep(sleep_t)
            else:
                next_t = time.time() # 串口或解算卡顿超时，强制重置基准时钟

    except KeyboardInterrupt:
        pass
    finally:
        # ==========================================
        # 9. 善后收尾与防断电重摔摔骨折保护
        # ==========================================
        print("[INFO] stopping...")
        try:
            # 命令全车所有电机用 0.6 秒时间平稳降落回到出厂最稳定的贴地姿态
            hwi.set_position_all(hwi.init_pos)
            time.sleep(0.6)
        except Exception:
            pass
        try:
            pb.disconnect() # 注销关闭弹出的 GUI 滑块面板
        except Exception:
            pass
        # 关闭所有舵机的力矩输出（彻底断电进入面条模式）
        hwi.turn_off()
        print("[INFO] torque disabled")


if __name__ == "__main__":
    main()