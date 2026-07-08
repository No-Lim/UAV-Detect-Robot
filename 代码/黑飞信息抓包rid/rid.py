#!/usr/bin/env python3
"""
RID 无人机扫描服务 — 运行在 RK3588 dogm 环境，供机器人控制 / YOLO 视觉读取。
用法: python rid_receiver.py [--port /dev/ttyUSB0] [--baud 115200]
输出: /tmp/rid_state.json  (当前活跃无人机快照，每秒更新)
"""

import json
import os
import signal
import sys
import tempfile
import time
import traceback
from datetime import datetime


# ═══════════════════════════════════════════════════════════════
# 配置
# ═══════════════════════════════════════════════════════════════

SERIAL_PORT    = os.environ.get("RID_PORT", "/dev/ttyUSB0")
SERIAL_BAUD    = int(os.environ.get("RID_BAUD", "115200"))
STATE_FILE     = os.environ.get("RID_STATE", "/tmp/rid_state.json")
WRITE_INTERVAL = float(os.environ.get("RID_WRITE_MS", "200")) / 1000.0  # 默认 200ms

STATUS_CN = {
    "undeclared": "未声明", "ground": "地面", "airborne": "空中",
    "emergency": "紧急", "rid_system_failure": "RID故障",
}

UA_TYPE_CN = {
    "none": "未知", "aeroplane": "固定翼",
    "helicopter_or_multirotor": "多旋翼", "gyroplane": "旋翼机",
    "hybrid_lift": "混合升力", "ornithopter": "扑翼机", "glider": "滑翔机",
    "free_balloon": "自由气球", "captive_balloon": "系留气球",
    "airship": "飞艇", "free_fall_parachute": "降落伞", "rocket": "火箭",
    "tethered_powered": "系留动力", "ground_obstacle": "地面障碍物", "other": "其他",
}


# ═══════════════════════════════════════════════════════════════
# 无人机状态存储
# ═══════════════════════════════════════════════════════════════

class DroneStore:
    """线程不安全但这里只有单线程，够用。"""

    def __init__(self):
        self._drones = {}   # mac -> dict

    def upsert(self, data):
        mac = data.get("mac", "")
        bid = data.get("basic_id", {}) or {}
        loc = data.get("location", {}) or {}
        sys_info = data.get("system", {}) or {}

        lat = loc.get("latitude", 0)
        lon = loc.get("longitude", 0)

        drone = {
            "mac": mac,
            "uas_id": bid.get("uas_id", ""),
            "ua_type": UA_TYPE_CN.get(bid.get("ua_type", ""), bid.get("ua_type", "")),
            "id_type": bid.get("id_type", ""),
            "status": STATUS_CN.get(loc.get("status", ""), loc.get("status", "")),
            "status_raw": loc.get("status", ""),
            "latitude": lat,
            "longitude": lon,
            "alt_baro": loc.get("alt_baro", 0),
            "alt_geo": loc.get("alt_geo", 0),
            "height": loc.get("height", 0),
            "speed_ms": loc.get("speed_h", 0),
            "speed_kmh": round(loc.get("speed_h", 0) * 3.6, 1),
            "speed_v_ms": loc.get("speed_v", 0),
            "direction": loc.get("direction", 0),
            "operator_lat": sys_info.get("operator_lat", 0),
            "operator_lon": sys_info.get("operator_lon", 0),
            "operator_alt": sys_info.get("operator_alt_geo", 0),
            "operator_loc_type": sys_info.get("operator_loc_type", ""),
            "rssi": data.get("rssi", 0),
            "channel": data.get("channel", 0),
            "protocol": data.get("protocol", ""),
            "msg_count": data.get("msg_count", 0),
            "first_seen": self._drones.get(mac, {}).get("first_seen", time.time()),
            "last_seen": time.time(),
        }
        self._drones[mac] = drone

    def remove(self, mac):
        self._drones.pop(mac, None)

    def cleanup(self, timeout=30):
        now = time.time()
        stale = [mac for mac, d in self._drones.items() if now - d["last_seen"] > timeout]
        for mac in stale:
            self._drones.pop(mac, None)
        return len(stale)

    def active_list(self):
        return sorted(self._drones.values(), key=lambda d: d["last_seen"], reverse=True)

    @property
    def count(self):
        return len(self._drones)


# ═══════════════════════════════════════════════════════════════
# 串口读取
# ═══════════════════════════════════════════════════════════════

def serial_lines(port, baud, shutdown):
    import serial
    while not shutdown[0]:
        try:
            ser = serial.Serial(port, baud, timeout=0.1)
            buf = b""
            while not shutdown[0]:
                chunk = ser.read(512)
                if chunk:
                    buf += chunk
                    while b"\n" in buf:
                        line, buf = buf.split(b"\n", 1)
                        yield line.decode("utf-8", errors="replace").strip()
        except (serial.SerialException, OSError) as e:
            if not shutdown[0]:
                print(f"[RID] 串口断开: {e}，2秒后重连...", flush=True)
                time.sleep(2)


def write_state(drones, path):
    """原子写入，避免读进程拿到半截数据。"""
    payload = json.dumps({
        "count": drones.count,
        "drones": drones.active_list(),
        "updated_at": datetime.now().isoformat(),
    }, ensure_ascii=False)

    fd, tmp = tempfile.mkstemp(dir=os.path.dirname(path), prefix=".rid_")
    try:
        os.write(fd, payload.encode("utf-8"))
        os.fsync(fd)
    finally:
        os.close(fd)
    os.replace(tmp, path)


# ═══════════════════════════════════════════════════════════════
# 主循环
# ═══════════════════════════════════════════════════════════════

def main():
    print(f"[RID] 启动 — 串口={SERIAL_PORT} 波特率={SERIAL_BAUD} 状态文件={STATE_FILE}", flush=True)

    store = DroneStore()
    shutdown = [False]

    def on_signal(sig, frame):
        shutdown[0] = True
        print(f"\n[RID] 收到信号 {sig}，正在退出...", flush=True)

    signal.signal(signal.SIGINT, on_signal)
    signal.signal(signal.SIGTERM, on_signal)

    last_write = 0.0
    last_cleanup = 0.0
    last_status = 0.0

    packet_count = 0
    rid_count = 0

    for line in serial_lines(SERIAL_PORT, SERIAL_BAUD, shutdown):
        if shutdown[0]:
            break

        if not line:
            continue

        # ESP32 的中文显示行跳过，不浪费解析
        if line.startswith("[ZH]"):
            continue

        try:
            data = json.loads(line)
        except json.JSONDecodeError:
            continue

        evt = data.get("evt", "")
        now = time.time()

        if evt == "uav_update" or evt == "uav_discovery":
            store.upsert(data)
            rid_count += 1

            if evt == "uav_discovery":
                uas_id = data.get("basic_id", {}).get("uas_id", "?")
                ua_type = UA_TYPE_CN.get(data.get("basic_id", {}).get("ua_type", ""), "")
                rssi = data.get("rssi", 0)
                print(f"[RID] 发现无人机: {uas_id} ({ua_type}) 信号={rssi}dBm", flush=True)

        elif evt == "uav_timeout":
            mac = data.get("mac", "")
            store.remove(mac)
            print(f"[RID] 无人机超时: {mac}", flush=True)

        elif evt == "status":
            store.cleanup(timeout=30)
            last_cleanup = now
            last_status = now
            active = data.get("active_uavs", store.count)
            pps = data.get("pkts_per_sec", 0)
            print(f"[RID] 状态刷新 | 在线{active}架 | {pps:.0f}包/秒", flush=True)

        packet_count += 1

        # 每 WRITE_INTERVAL 写一次状态文件；每 10 秒清理超时
        if now - last_write >= WRITE_INTERVAL:
            write_state(store, STATE_FILE)
            last_write = now

        if now - last_cleanup >= 10.0:
            cleaned = store.cleanup(timeout=30)
            if cleaned:
                print(f"[RID] 清理 {cleaned} 架超时无人机", flush=True)
            last_cleanup = now

    # 退出前清空状态文件
    print("[RID] 服务已停止", flush=True)
    try:
        os.remove(STATE_FILE)
    except FileNotFoundError:
        pass


if __name__ == "__main__":
    main()
