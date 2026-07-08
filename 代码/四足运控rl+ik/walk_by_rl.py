#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import time
import pickle
import numpy as np
import os
import sys

# ==========================================
# 1. 动态调整 Python 搜索路径
# ==========================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# 从核心运行库 runtime 文件夹中被动借调所有的零部件
from runtime.position_hwi import HWI
from runtime.onnx_infer import OnnxInfer
from runtime.raw_imu import Imu
from runtime.xbox import XBoxController
from runtime.rl_utils import make_action_dict, LowPassActionFilter

HOME_DIR = os.path.expanduser("~")


class RLWalk:
    def __init__(
        self,
        onnx_model_path: str,
        serial_port: str = "/dev/ttyACM0",
        control_freq: float = 50,  # 实机闭环控制频率，默认为 50Hz (每 20ms 一个周期)
        pid=[30, 0, 0],             # 电机工作刚度，对应 [Kp, Ki, Kd]
        action_scale=0.25,          # 动作缩放系数：用于限制神经网络输出幅值，保护电机
        commands=False,             # 是否开启外部遥控（键盘/手柄）
        pitch_bias=0,               # IMU 静态俯仰角修正偏置
        cutoff_frequency=None,      # 低通滤波器的截止频率
    ):

        self.cmd_max = np.array([0.15, 0.2, 1.0])  # [前进m/s, 侧移m/s, 转向rad/s]
        self.commands = commands

        # 实例化 ONNX 策略网络推理会话
        self.onnx_model_path = onnx_model_path
        self.policy = OnnxInfer(self.onnx_model_path, awd=True)

        self.num_dofs = 12                # 机械狗 12 个自由度（关节）
        self.max_motor_velocity = 5.24    # 限制电机的最大极限转速 (rad/s)

        self.control_freq = control_freq
        self.pid = pid

        # 一阶低通动作滤波器初始化（如果开启，能有效去除 AI 输出的毛刺信号，让动作丝滑）
        self.action_filter = None
        if cutoff_frequency is not None:
            self.action_filter = LowPassActionFilter(
                self.control_freq, cutoff_frequency
            )

        # 建立 12 个关节总线的串口物理连接
        self.hwi = HWI(serial_port)

        # 安全上电，缓慢起立过渡到初始姿态
        self.start()

        # 启动后台 IMU 原始惯性量（角速度/加速度）采集线程
        self.imu = Imu(
            sampling_freq=int(self.control_freq)
        )

        self.action_scale = action_scale

        # 强化学习历史动作记录器（核心设计）：
        # 很多强化学习策略网络在训练时，都需要知道前 3 帧历史动作（History Actions）作为输入，
        # 用于协助神经网络隐式地推算出足端是否触地或机械结构的动力学惯性。
        self.last_action = np.zeros(self.num_dofs)
        self.last_last_action = np.zeros(self.num_dofs)
        self.last_last_last_action = np.zeros(self.num_dofs)

        # 从 HWI 中提取开机默认站立角度列表以及符号对齐列表
        self.init_pos = list(self.hwi.init_pos.values())
        self.joint_signs = list(self.hwi.real_pose_signs_rl.values()) 

        self.motor_targets = np.array(self.init_pos.copy())

        # 用户遥控速度目标指令缓存：[前进/后退速度, 左右平移速度, 原地打转角速度]
        self.last_commands = [0.0, 0.0, 0.0]

        self.paused = False         # 暂停开关
        self.command_freq = 20      # 遥控器捕获频率 (20Hz)
        if self.commands:
            self.xbox_controller = XBoxController(self.command_freq)

###对陀螺仪
###对关节角度 


    def get_obs(self):
        """
        状态观测值搜集中枢 (Get Observations):
        将全车所有的感知传感器数据打包在一起，构建神经网络要求的输入 Observation。
        """
        # 1. 抓取最新的 IMU 惯性量
        imu_data = self.imu.get_data()

        # 2. 索要全车 12 个关节当前的实际物理角度反馈 (rad)
        dof_pos = self.hwi.get_present_positions()

        # 3. 索要全车 12 个关节当前的实际转速反馈 (rad/s)
        dof_vel = self.hwi.get_present_velocities()

        # 安全防空机制：如果任何一个硬件发生通信延迟没读到数据，立刻放弃本帧控制，防止狗暴走
        if dof_pos is None or dof_vel is None:
            return None

        if len(dof_pos) != self.num_dofs or len(dof_vel) != self.num_dofs:
            print(f"ERROR: 关节反馈维度不等于 12")
            return None

        # 提取当前的用户操控指令向量
    
        cmds = np.asarray(self.last_commands, dtype=float)[:3] * self.cmd_max        
        # 核心坐标系变换（对齐仿真）：
        # 仿真环境（Isaac/MuJoCo）中，输入的往往是相对于初始站立姿态的【相对角度】。
        # 公式：相对角度 = (实机当前角度 - 初始站立参考角度) * 镜像方向符号
        dof_pos_rel = (dof_pos - self.init_pos) * self.joint_signs
        dof_vel_rl = dof_vel * self.joint_signs # 角速度也必须对齐物理镜像方向

        # 拼装终极状态大一维向量（严格按照仿真训练时的拼接顺序对齐！）
        obs = np.concatenate(
            [
                imu_data["gyro"],              # 3维：三轴角速度
                imu_data["accelero"],          # 3维：三轴线加速度
                cmds,                          # 3维：用户遥控速度指令
                dof_pos_rel,                   # 12维：各关节相对当前站立位的偏移角度
                dof_vel_rl * 0.05,             # 12维：各关节角速度，乘以 0.05 的缩放系数
                self.last_action,              # 12维：上一帧网络输出的 Action
                self.last_last_action,         # 12维：上上帧网络输出的 Action
                self.last_last_last_action,    # 12维：上上上帧网络输出的 Action
            ]
        )
        return obs

    def start(self):
        """ 开机引导与安全上电 """
        n = len(self.hwi.joints)
        kp = float(self.pid[0])
        kd = float(self.pid[2])

        # 初始化刷写电机的 PID 参数
        kps = [kp] * n
        kds = [kd] * n
        self.hwi.set_kps(kps)
        self.hwi.set_kds(kds)
        
        # 调用 HWI 的软上电逻辑：以低刚度慢速挪动到初始 init_pos 站立位，然后瞬间锁死，抵抗重力站立
        self.hwi.turn_on()
        time.sleep(1.0) # 撑住并稳定 1 秒钟

    def run(self):
        """ 50Hz 实机神经网络闭环控制大循环 (Real-time Inference Loop) """
        i = 0
        try:
            print("Starting RL Walk loop...")
            start_t = time.time()
            while True:
                t = time.time()

                # 4.1 捕获虚拟/物理手柄的用户指令
                if self.commands:
                    self.last_commands, self.buttons, _, _ = (
                        self.xbox_controller.get_last_command()
                    )
                    # 边沿触发检测：若按下空格（A键），一键切入/切出紧急暂停状态
                    if self.buttons.A.triggered:
                        self.paused = not self.paused
                        if self.paused:
                            print("=== 暂停控制 (PAUSE) ===")
                        else:
                            print("=== 恢复控制 (UNPAUSE) ===")

                # 如果处于暂停状态，锁定当前的关节，挂起等待
                if self.paused:
                    time.sleep(0.1)
                    continue

                # 4.2 采集传感器并构造状态空间 Observation
                obs = self.get_obs()
                if obs is None:
                    continue

                # 4.3 【核心动作】：喂给神经网络，进行毫秒级的高速前向推理
                action = np.asarray(self.policy.infer(obs), dtype=float)
                print(action)

                # 4.4 动作滚筒队列推进：缓存历史 3 帧动作，留给下一轮 get_obs 使用
                self.last_last_last_action = self.last_last_action.copy()
                self.last_last_action = self.last_action.copy()
                self.last_action = action.copy()

                # 4.5 神经网络相对 Action 转化为实机绝对目标角度指令
                # 公式：物理目标弧度 = 初始参考基准 + 神经网络动作值 * 缩放幅值 * 镜像符号
                self.motor_targets = (
                    self.init_pos + action * self.action_scale * self.joint_signs
                )

                # 4.6 动作低通滤波：磨掉突变棱角，防止实机机械关节高频剧烈震颤
                if self.action_filter is not None:
                    self.action_filter.push(self.motor_targets)
                    filtered_motor_targets = self.action_filter.get_filtered_action()
                    # 开机前 1 秒由于网络刚启动数据不稳，等待 1 秒稳定后再交由低通滤波器全权接管
                    if (time.time() - start_t > 1):
                        self.motor_targets = filtered_motor_targets

                # 4.7 将计算好的一维角度数组转换为带关节名称的命令字典
                action_dict = make_action_dict(
                    self.motor_targets, list(self.hwi.joints.keys())
                )
                print(action_dict)
                # 4.8 最终通过串行总线把 12 个目标位置一键下发给硬件肌肉执行
                self.hwi.set_position_all(action_dict)
                i += 1
                
                # ==========================================
                # 5. 严格的时间控制与超速死区防御
                # ==========================================
                took = time.time() - t # 计算本轮数据交互+推理总共耗时了多少秒
                
                # 警告防摔机制：如果发现本周期的计算开销超出了 50Hz 允许的 20ms 时间预算（控制预算超标），
                # 说明总线卡顿、或者是 CPU 严重降频，立刻打印警告。
                if (1 / self.control_freq - took) < 0:
                    print(
                        "Policy control budget exceeded by",
                        np.around(took - 1 / self.control_freq, 3), "seconds!"
                    )
                # 动态计算休眠时间，强行让主循环稳定保持在完美的 50Hz 闭环控制
                time.sleep(max(0, 1 / self.control_freq - took))

        except KeyboardInterrupt:
            # 捕获随时可能发生的终端 Ctrl+C
            pass
        finally:
            # ==========================================
            # 6. 安全善后与停机卸力逻辑
            # ==========================================
            print("\nTURNING OFF AND CLEANING UP...")
            try:
                if self.commands and hasattr(self, "xbox_controller"):
                    self.xbox_controller.close() # 关闭虚拟终端键盘控制模式，还原配置
            except Exception as e:
                print("Failed to close controller:", e)

            try:
                self.hwi.turn_off() # 强行切断全身 12 个电机的输出扭矩（释放肌肉），保护硬件不发热
                print("[SUCCESS] 全车电机已成功切断力矩，处于放松模式。")
            except Exception as e:
                print("Failed to turn off motors:", e)


if __name__ == "__main__":
    import argparse

    # 命令行参数解析：支持在终端直接指定不同的权重模型文件或调整刚度
    parser = argparse.ArgumentParser(description="四足机器人强化学习实机全自动部署主入口")
    parser.add_argument("--onnx_model_path", type=str, default="/home/elf/Desktop/Learn-It-All-deployment-sim2real-main/best.onnx")
    parser.add_argument("-a", "--action_scale", type=float, default=0.25)
    parser.add_argument("-p", type=int, default=30, help="位置比例增益 Kp")
    parser.add_argument("-i", type=int, default=0)
    parser.add_argument("-d", type=int, default=0, help="微分阻尼增益 Kd")
    parser.add_argument("-c", "--control_freq", type=int, default=50) # 推理闭环控制频率
    parser.add_argument("--pitch_bias", type=float, default=0, help="单位:度(deg)")
    parser.add_argument(
        "--commands",
        action="store_true",
        default=True,
        help="开启外部控制。如果加此开关，可以在电脑上启动控制服务器进行跨网络传输键盘信号",
    )
    # 如果遇到电机发热或高频滋滋颤抖，建议在终端追加：`--cutoff_frequency 10.0`
    parser.add_argument("--cutoff_frequency", type=float, default=None)

    args = parser.parse_args()
    pid = [args.p, args.i, args.d]

    print("Done parsing args")
    # 初始化控制循环大类
    rl_walk = RLWalk(
        args.onnx_model_path,
        action_scale=args.action_scale,
        pid=pid,
        control_freq=args.control_freq,
        commands=args.commands,
        pitch_bias=args.pitch_bias,
        cutoff_frequency=args.cutoff_frequency,
    )
    print("Done instantiating RLWalk")
    
    # 彻底引爆主循环，机械狗开始通过 RL 决策向前奔跑！
    rl_walk.run()