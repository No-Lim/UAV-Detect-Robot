# coding:UTF-8
"""
    JY901 IMU 延时节流打印测试文件
"""
import time
import datetime
import platform
import struct

# pip全局安装witimu包，直接顶层导入lib
import lib.device_model as deviceModel
from lib.data_processor.roles.jy901s_dataProcessor import JY901SDataProcessor
from lib.protocol_resolver.roles.wit_protocol_resolver import WitProtocolResolver

welcome = """
欢迎使用维特智能示例程序    Welcome to the Wit-Motoin sample program
"""
_writeF = None                    # 文件句柄
_IsWriteF = False                 # 写入开关
last_print_time = 0               # 打印节流时间戳

def readConfig(device):
    """读取设备配置寄存器"""
    tVals = device.readReg(0x02,3)
    if len(tVals) > 0:
        print("寄存器0x02返回：" + str(tVals))
    else:
        print("无返回数据")
    tVals = device.readReg(0x23,2)
    if len(tVals) > 0:
        print("寄存器0x23返回：" + str(tVals))
    else:
        print("无返回数据")

def setConfig(device):
    """修改设备配置"""
    device.unlock()
    time.sleep(0.1)
    device.writeReg(0x03, 6)       # 10HZ回传
    time.sleep(0.1)
    device.writeReg(0x23, 0)
    time.sleep(0.1)
    device.writeReg(0x24, 0)
    time.sleep(0.1)
    device.save()

def AccelerationCalibration(device):
    """加速度校准"""
    device.AccelerationCalibration()
    print("加计校准完成")

def FiledCalibration(device):
    """磁场校准"""
    device.BeginFiledCalibration()
    if input("三轴缓慢转圈完毕，输入Y结束校准：").lower() == "y":
        device.EndFiledCalibration()
        print("磁场校准完成")

def onUpdate(deviceModel):
    global last_print_time
    now = time.time()
    print_interval = 0.5  # 打印间隔 0.5秒，可自行改大小

    # 无论打不打印，只要开启记录就写入文件
    if _IsWriteF:
        Tempstr = " " + str(deviceModel.getDeviceData("Chiptime"))
        Tempstr += "\t"+str(deviceModel.getDeviceData("accX")) + "\t"+str(deviceModel.getDeviceData("accY"))+"\t"+ str(deviceModel.getDeviceData("accZ"))
        Tempstr += "\t" + str(deviceModel.getDeviceData("gyroX")) +"\t"+ str(deviceModel.getDeviceData("gyroY")) +"\t"+ str(deviceModel.getDeviceData("gyroZ"))
        Tempstr += "\t" + str(deviceModel.getDeviceData("angleX")) +"\t" + str(deviceModel.getDeviceData("angleY")) +"\t"+ str(deviceModel.getDeviceData("angleZ"))
        Tempstr += "\t" + str(deviceModel.getDeviceData("temperature"))
        Tempstr += "\t" + str(deviceModel.getDeviceData("magX")) +"\t" + str(deviceModel.getDeviceData("magY")) +"\t"+ str(deviceModel.getDeviceData("magZ"))
        Tempstr += "\t" + str(deviceModel.getDeviceData("lon")) + "\t" + str(deviceModel.getDeviceData("lat"))
        Tempstr += "\t" + str(deviceModel.getDeviceData("Yaw")) + "\t" + str(deviceModel.getDeviceData("Speed"))
        Tempstr += "\t" + str(deviceModel.getDeviceData("q1")) + "\t" + str(deviceModel.getDeviceData("q2"))
        Tempstr += "\t" + str(deviceModel.getDeviceData("q3")) + "\t" + str(deviceModel.getDeviceData("q4"))
        Tempstr += "\r\n"
        _writeF.write(Tempstr)

    # 节流控制打印频率
    if now - last_print_time < print_interval:
        return
    last_print_time = now

    # 精简格式化打印，可读性更强
    ax = round(deviceModel.getDeviceData("accX"), 2)
    ay = round(deviceModel.getDeviceData("accY"), 2)
    az = round(deviceModel.getDeviceData("accZ"), 2)
    gx = round(deviceModel.getDeviceData("gyroX"), 2)
    gy = round(deviceModel.getDeviceData("gyroY"), 2)
    gz = round(deviceModel.getDeviceData("gyroZ"), 2)
    roll = round(deviceModel.getDeviceData("angleX"), 2)
    pitch = round(deviceModel.getDeviceData("angleY"), 2)
    yaw = round(deviceModel.getDeviceData("angleZ"), 2)

    print(f"【姿态】Roll:{roll:6.2f}° Pitch:{pitch:6.2f}° Yaw:{yaw:6.2f}° | 加速度 X:{ax:5.2f} Y:{ay:5.2f} Z:{az:5.2f}")

def startRecord():
    global _writeF
    global _IsWriteF
    filename = datetime.datetime.now().strftime('%Y%m%d%H%M%S') + ".txt"
    _writeF = open(filename, "w", encoding="utf-8")
    _IsWriteF = True
    # 写入表头
    header = "Chiptime\tax(g)\tay(g)\taz(g)\twx(deg/s)\twy(deg/s)\twz(deg/s)\tAngleX(deg)\tAngleY(deg)\tAngleZ(deg)\tT(°)\tmagx\tmagy\tmagz\tlon\tlat\tYaw\tSpeed\tq1\tq2\tq3\tq4\r\n"
    _writeF.write(header)
    print(f"已开启数据记录，保存文件名：{filename}")

def endRecord():
    global _writeF
    global _IsWriteF
    _IsWriteF = False
    _writeF.close()
    print("已关闭数据记录，文件保存完成")

if __name__ == '__main__':
    print(welcome)
    # 初始化设备实例
    device = deviceModel.DeviceModel(
        "我的JY901",
        WitProtocolResolver(),
        JY901SDataProcessor(),
        "51_0"
    )

    # 串口配置
    if platform.system().lower() == 'linux':
        device.serialConfig.portName = "/dev/ttyUSB0"
    else:
        device.serialConfig.portName = "COM17"
    device.serialConfig.baud = 921600

    # 打开串口
    device.openDevice()
    print("串口打开成功")
    readConfig(device)

    # 绑定数据回调
    device.dataProcessor.onVarChanged.append(onUpdate)

    try:
        print("\n程序持续接收IMU数据，Ctrl+C终止程序")
        while True:
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\n检测到退出指令，关闭资源")
        if _IsWriteF:
            endRecord()
        device.closeDevice()
        print("设备串口已关闭，程序退出")