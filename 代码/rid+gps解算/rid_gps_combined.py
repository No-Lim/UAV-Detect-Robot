#!/usr/bin/env python3
"""
RID + GPS 合并版 — 双线程读取 ESP32 和 GPS 串口，实时解算无人机相对机器狗位置。
用法: python rid_gps_combined.py
输出: /tmp/rid_state.json (每 200ms 更新)
"""

import json
import math
import os
import signal
import sys
import tempfile
import threading
import time
from datetime import datetime, timezone


# ═══════════════════════════════════════════════════════════════
# 配置
# ═══════════════════════════════════════════════════════════════

RID_PORT       = os.environ.get("RID_PORT", "/dev/ttyUSB0")
RID_BAUD       = int(os.environ.get("RID_BAUD", "115200"))
GPS_PORT       = os.environ.get("GPS_PORT", "/dev/ttyUSB2")
GPS_BAUD       = int(os.environ.get("GPS_BAUD", "9600"))
STATE_FILE     = os.environ.get("RID_STATE", "/tmp/rid_state.json")
WRITE_INTERVAL = float(os.environ.get("RID_WRITE_MS", "200")) / 1000.0

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

FIX_QUALITY_CN = {0: "无定位", 1: "GPS", 2: "DGPS", 4: "RTK固定", 5: "RTK浮点"}


# ═══════════════════════════════════════════════════════════════
# 共享 GPS 状态 (GPS 线程写，主线程读，加锁)
# ═══════════════════════════════════════════════════════════════

gps_lock = threading.Lock()
gps_state = {
    "lat": 0.0, "lon": 0.0, "alt_m": 0.0,
    "speed_ms": 0.0, "track": 0.0,
    "satellites": 0, "hdop": 99.9, "fix_quality": 0,
    "has_fix": False, "last_fix_ts": 0.0,
}


def get_gps():
    """线程安全地读取当前 GPS 快照"""
    with gps_lock:
        return dict(gps_state)


# ═══════════════════════════════════════════════════════════════
# 位置解算
# ═══════════════════════════════════════════════════════════════

def haversine_distance(lat1, lon1, lat2, lon2):
    """Haversine 公式计算两点距离 (米)"""
    R = 6371000
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat / 2) ** 2 +
         math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) *
         math.sin(dlon / 2) ** 2)
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def bearing_to(lat1, lon1, lat2, lon2):
    """从点1到点2的方位角 (度, 0=北, 90=东)"""
    dlon = math.radians(lon2 - lon1)
    y = math.sin(dlon) * math.cos(math.radians(lat2))
    x = (math.cos(math.radians(lat1)) * math.sin(math.radians(lat2)) -
         math.sin(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.cos(dlon))
    return (math.degrees(math.atan2(y, x)) + 360) % 360


def compute_relative(drone_lat, drone_lon, drone_alt):
    """算无人机相对机器狗的 距离/方位/高度差。GPS 未定位返回 None。"""
    gps = get_gps()
    if not gps["has_fix"] or drone_lat == 0 or drone_lon == 0:
        return None
    dist = haversine_distance(gps["lat"], gps["lon"], drone_lat, drone_lon)
    az = bearing_to(gps["lat"], gps["lon"], drone_lat, drone_lon)
    dalt = drone_alt - gps["alt_m"]
    return {
        "distance_m": round(dist, 1),
        "azimuth_deg": round(az, 1),
        "delta_alt_m": round(dalt, 1),
    }


# ═══════════════════════════════════════════════════════════════
# NMEA 解析 (GPS 线程)
# ═══════════════════════════════════════════════════════════════

def nmea_checksum_ok(sentence):
    if '*' not in sentence:
        return False
    data, _, cs = sentence.partition('*')
    if not cs:
        return False
    try:
        expected = int(cs.strip(), 16)
    except ValueError:
        return False
    xor = 0
    for ch in data:
        xor ^= ord(ch)
    return xor == expected


def parse_dm(raw, hemi):
    if not raw or '.' not in raw:
        return 0.0
    dot = raw.index('.')
    if dot < 3:
        return 0.0
    deg = float(raw[:dot - 2])
    minutes = float(raw[dot - 2:])
    val = deg + minutes / 60.0
    if hemi in ('S', 'W'):
        val = -val
    return round(val, 8)


# ═══════════════════════════════════════════════════════════════
# GPS 读取线程
# ═══════════════════════════════════════════════════════════════

def gps_thread(port, baud, shutdown):
    import serial
    print(f"[GPS] 线程启动 — {port} {baud}", flush=True)

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
                        process_nmea(line.decode("ascii", errors="replace").strip())
        except (serial.SerialException, OSError) as e:
            if not shutdown[0]:
                print(f"[GPS] 串口断开: {e}，2秒后重连...", flush=True)
                time.sleep(2)

    print("[GPS] 线程退出", flush=True)


def process_nmea(line):
    global gps_state
    if not line or len(line) < 8 or line[0] != '$':
        return
    if not nmea_checksum_ok(line[1:]):
        return

    fields = line[1:].split(',')
    talker = fields[0]

    if talker in ('GPRMC', 'GNRMC'):
        if len(fields) < 10 or fields[2] != 'A':
            return
        lat = parse_dm(fields[3], fields[4])
        lon = parse_dm(fields[5], fields[6])
        if lat == 0 and lon == 0:
            return
        speed_kn = float(fields[7] or 0)
        track = float(fields[8] or 0)

        with gps_lock:
            was_fixed = gps_state["has_fix"]
            gps_state["lat"] = lat
            gps_state["lon"] = lon
            gps_state["speed_ms"] = round(speed_kn * 0.514444, 2)
            gps_state["track"] = round(track, 1)
            gps_state["last_fix_ts"] = time.time()
            gps_state["has_fix"] = True
            if not was_fixed:
                print(f"[GPS] 定位成功! ({lat:.6f}, {lon:.6f})", flush=True)

    elif talker in ('GPGGA', 'GNGGA'):
        if len(fields) < 10:
            return
        quality = int(fields[6] or 0)
        if quality == 0:
            return
        alt = float(fields[9] or 0)
        with gps_lock:
            gps_state["alt_m"] = round(alt, 1)
            gps_state["fix_quality"] = quality
            gps_state["satellites"] = int(fields[7] or 0)
            gps_state["hdop"] = round(float(fields[8] or 99.9), 1)


# ═══════════════════════════════════════════════════════════════
# RID 串口读取 (主线程)
# ═══════════════════════════════════════════════════════════════

def rid_lines(port, baud, shutdown):
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


def atomic_write(path, data):
    fd, tmp = tempfile.mkstemp(dir=os.path.dirname(path), prefix=".rid_")
    try:
        os.write(fd, json.dumps(data, ensure_ascii=False).encode("utf-8"))
        os.fsync(fd)
    finally:
        os.close(fd)
    os.replace(tmp, path)


# ═══════════════════════════════════════════════════════════════
# 无人机存储
# ═══════════════════════════════════════════════════════════════

class DroneStore:
    def __init__(self):
        self._drones = {}

    def upsert(self, data):
        mac = data.get("mac", "")
        bid = data.get("basic_id", {}) or {}
        loc = data.get("location", {}) or {}
        sys_info = data.get("system", {}) or {}

        lat = loc.get("latitude", 0)
        lon = loc.get("longitude", 0)
        alt = loc.get("alt_baro", 0)

        rel = compute_relative(lat, lon, alt)

        drone = {
            "mac": mac,
            "uas_id": bid.get("uas_id", ""),
            "ua_type": UA_TYPE_CN.get(bid.get("ua_type", ""), bid.get("ua_type", "")),
            "status": STATUS_CN.get(loc.get("status", ""), loc.get("status", "")),
            "lat": lat,
            "lon": lon,
            "alt_m": alt,
            "speed_kmh": round(loc.get("speed_h", 0) * 3.6, 1),
            "dir_deg": loc.get("direction", 0),
            "rssi": data.get("rssi", 0),
            "dist_m": rel["distance_m"] if rel else None,
            "azimuth_deg": rel["azimuth_deg"] if rel else None,
            "delta_alt_m": rel["delta_alt_m"] if rel else None,
            "op_lat": sys_info.get("operator_lat", 0),
            "op_lon": sys_info.get("operator_lon", 0),
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
# 主函数
# ═══════════════════════════════════════════════════════════════

def main():
    print(f"[RID+GPS] 启动", flush=True)
    print(f"  ESP32 → {RID_PORT} @ {RID_BAUD}", flush=True)
    print(f"  GPS   → {GPS_PORT} @ {GPS_BAUD}", flush=True)
    print(f"  输出  → {STATE_FILE}", flush=True)

    store = DroneStore()
    shutdown = [False]

    def on_signal(sig, frame):
        shutdown[0] = True
        print(f"\n[RID+GPS] 收到信号 {sig}，正在退出...", flush=True)

    signal.signal(signal.SIGINT, on_signal)
    signal.signal(signal.SIGTERM, on_signal)

    # 启动 GPS 线程
    gps_t = threading.Thread(target=gps_thread, args=(GPS_PORT, GPS_BAUD, shutdown), daemon=True)
    gps_t.start()

    last_write = 0.0
    last_cleanup = 0.0
    last_status = 0.0

    for line in rid_lines(RID_PORT, RID_BAUD, shutdown):
        if shutdown[0]:
            break

        if not line or line.startswith("[ZH]"):
            continue

        try:
            data = json.loads(line)
        except json.JSONDecodeError:
            continue

        evt = data.get("evt", "")
        now = time.time()

        if evt in ("uav_update", "uav_discovery"):
            store.upsert(data)

            if evt == "uav_discovery":
                uas_id = data.get("basic_id", {}).get("uas_id", "?")
                ua_type = UA_TYPE_CN.get(data.get("basic_id", {}).get("ua_type", ""), "")
                rssi = data.get("rssi", 0)
                d = store._drones.get(data.get("mac", ""), {})
                rel_str = f" 距离{d['dist_m']:.0f}m 方位{d['azimuth_deg']:.0f}°" if d.get("dist_m") is not None else ""
                print(f"[RID] 发现无人机: {uas_id} ({ua_type}) 信号={rssi}dBm{rel_str}", flush=True)

        elif evt == "uav_timeout":
            store.remove(data.get("mac", ""))

        elif evt == "status":
            cleaned = store.cleanup(timeout=30)
            last_cleanup = now
            gps = get_gps()
            fix_label = FIX_QUALITY_CN.get(gps["fix_quality"], "?")
            gps_str = f"GPS({gps['lat']:.6f},{gps['lon']:.6f}) {fix_label} 卫星{gps['satellites']}" if gps["has_fix"] else "GPS未定位"
            print(f"[RID] 状态 | 在线{store.count}架 | {gps_str}", flush=True)

        # 定时写出
        if now - last_write >= WRITE_INTERVAL:
            gps = get_gps()
            payload = {
                "count": store.count,
                "drones": store.active_list(),
                "gps": {
                    "lat": gps["lat"],
                    "lon": gps["lon"],
                    "alt_m": gps["alt_m"],
                    "satellites": gps["satellites"],
                    "has_fix": gps["has_fix"],
                },
                "updated_at": datetime.now().isoformat(),
            }
            atomic_write(STATE_FILE, payload)
            last_write = now

        # 定时清理
        if now - last_cleanup >= 10.0:
            cleaned = store.cleanup(timeout=30)
            if cleaned:
                print(f"[RID] 清理 {cleaned} 架超时无人机", flush=True)
            last_cleanup = now

    # 退出
    print("[RID+GPS] 服务已停止", flush=True)
    try:
        os.remove(STATE_FILE)
    except FileNotFoundError:
        pass


if __name__ == "__main__":
    main()
