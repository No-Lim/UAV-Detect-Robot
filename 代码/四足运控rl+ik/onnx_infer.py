#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import onnxruntime


class OnnxInfer:
    def __init__(self, onnx_model_path, input_name="obs", awd=False):
        """
        ONNX 模型推理加速类
        :param onnx_model_path: ONNX 模型文件路径 (.onnx)
        :param input_name: 模型输入节点的名称，默认四足机器人中常用 "obs" (Observations，环境观测值)
        :param awd: Auto-Wrap Dimension 缩写，是否自动为输入数据包裹一层 Batch 维度
        """
        self.onnx_model_path = onnx_model_path
        
        # ==========================================
        # 1. 实例化 ONNX 推理会话 (InferenceSession)
        # ==========================================
        # 指定使用 CPU 进行推理 (CPUExecutionProvider)，适合边缘计算设备如树莓派、边缘主控等
        self.ort_session = onnxruntime.InferenceSession(
            self.onnx_model_path, providers=["CPUExecutionProvider"]
        )
        self.input_name = input_name
        self.awd = awd

    def infer(self, inputs):
        """
        执行前向推理
        :param inputs: 输入的观测数据 (通常为 1D 数组或 2D 矩阵)
        :return: 模型的输出（在四足机器人中通常是 12 个关节的目标角度/动作）
        """
        if self.awd:
            # ==========================================
            # 2. 自动升维推理分支 (awd=True)
            # ==========================================
            # 如果传入的 inputs 是一个一维向量（例如没有 Batch 维度的单帧数据），
            # 通过 `[inputs]` 将其包装为二维输入，模拟 `shape = (1, N)` 的效果，满足神经网络对 Batch 维度的要求。
            outputs = self.ort_session.run(None, {self.input_name: [inputs]})
            
            # 由于输入包了一层 [inputs]，输出也会多一层，所以通过 [0][0] 剥离出实际的 1D 输出结果
            return outputs[0][0]
        else:
            # ==========================================
            # 3. 标准推理分支 (awd=False)
            # ==========================================
            # 此时要求传入的 inputs 已经具备正确的维度，并在此强制转换为 float32 类型
            outputs = self.ort_session.run(
                None, {self.input_name: inputs.astype("float32")}
            )
            # 返回原生输出列表中第一个输出节点的结果
            return outputs[0]


if __name__ == "__main__":
    import argparse
    import numpy as np
    import time

    # ==========================================
    # 4. 命令行参数解析
    # ==========================================
    parser = argparse.ArgumentParser(description="ONNX 推理性能测试工具")
    parser.add_argument("-o", "--onnx_model_path", type=str, required=True, help="ONNX 模型文件路径")
    args = parser.parse_args()

    # 初始化推理类，开启自动包装 Batch 维度 (awd=True)
    oi = OnnxInfer(args.onnx_model_path, awd=True)
    
    # 构造测试输入数据：
    # 1. 模拟随机数据（大小为 54，这在很多四足机器人的强化学习观测空间中很常见，例如历史状态堆叠）
    inputs = np.random.uniform(size=54).astype(np.float32)
    # 2. 覆盖上一行，改用 0 到 46 连续递增的 47 维测试向量（具体维度取决于你的 Policy 策略网络输入定义）
    inputs = np.arange(47).astype(np.float32)
    
    times = []
    
    # ==========================================
    # 5. 性能基准测试循环 (Benchmark)
    # ==========================================
    print("开始执行 1000 次推理基准测试...")
    for i in range(1000):
        start = time.time()
        
        # 执行推理并打印每次网络输出的动作指令（如舵机目标角度）
        print(oi.infer(inputs))
        
        # 记录单次推理耗时（秒）
        times.append(time.time() - start)

    # ==========================================
    # 6. 输出性能指标
    # ==========================================
    avg_time = sum(times) / len(times)
    print("\n" + "="*40)
    print(f"平均单次推理耗时 (Average time): {avg_time * 1000:.3f} ms")
    print(f"平均每秒推理次数 (Average FPS) : {1 / avg_time:.1f} Hz")
    print("="*40)