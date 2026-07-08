#!/usr/bin/env python3
"""
Red block tracking + QGimbal control.

This keeps the same gimbal control strategy as dect.py, but replaces YOLO
detection with OpenCV HSV red color tracking.
"""

import argparse
import glob
import json
import os
import re
import struct
import tempfile
import time

import cv2
import numpy as np
import serial
from rknnlite.api import RKNNLite


CAMERA_SOURCE = '/dev/video21'
DET_FILE = '/tmp/yolo_detections.json'
GIMBAL_PORT = '/dev/ttyUSB1'
RKNN_MODEL = '/home/elf/Desktop/rknn/last.rknn'
CLASSES = ("UAV",)
OBJ_THRESH = 0.25
NMS_THRESH = 0.25
IMG_SIZE = (640, 640)

STRIDES = (8, 16, 32)
GRIDS = {}
for stride in STRIDES:
    gh, gw = IMG_SIZE[1] // stride, IMG_SIZE[0] // stride
    xv, yv = np.meshgrid(np.arange(gw), np.arange(gh))
    GRIDS[stride] = np.stack([xv, yv], axis=-1).reshape(-1, 2).astype(np.float32)
DFL_PROJ = np.arange(16, dtype=np.float32)

CMD_NOP = 0x00
CMD_ENABLE = 0x01
CMD_DISABLE = 0x02
CMD_SPEED = 0x04
CMD_ANGLE = 0x05
CMD_LOW_SPEED = 0x06
CMD_LASER_OFF = 0xFC
CMD_LASER_ON = 0xFD
CMD_STABILITY_ON = 0xFF


def crc8(data):
    crc = 0
    for byte in data:
        crc ^= byte
        for _ in range(8):
            if crc & 0x80:
                crc = ((crc << 1) ^ 0x07) & 0xFF
            else:
                crc = (crc << 1) & 0xFF
    return crc


def make_gimbal_packet(cmd, yaw=0.0, pitch=0.0):
    payload = struct.pack('<Bff', cmd, float(yaw), float(pitch))
    return payload + bytes([crc8(payload)])


def parse_gimbal_feedback(packet):
    if len(packet) != 42:
        raise ValueError(f'feedback packet must be 42 bytes, got {len(packet)}')
    if crc8(packet[:41]) != packet[41]:
        raise ValueError('feedback CRC check failed')
    values = struct.unpack('<B10f', packet[:41])
    status = values[0]
    return {
        'enabled': bool(status & 0x01),
        'stabilized': bool(status & 0x02),
        'laser': bool(status & 0x04),
        'yaw_motor_rad': values[9],
        'pitch_motor_rad': values[10],
    }


def read_feedback_from_serial(ser):
    ser.reset_input_buffer()
    ser.write(make_gimbal_packet(CMD_NOP))
    data = ser.read(128)
    for offset in range(0, len(data) - 41):
        packet = data[offset:offset + 42]
        try:
            return parse_gimbal_feedback(packet)
        except ValueError:
            continue
    return None


class GimbalController:
    def __init__(self, port, baud, protocol, yaw_kp, pitch_kp, yaw_kd, pitch_kd, max_yaw, max_pitch,
                 deadzone, yaw_sign, pitch_sign, control_mode, max_yaw_step,
                 max_pitch_step, laser_on_target, laser_deadzone, target_offset_x,
                 target_offset_y):
        self.ser = serial.Serial(port, baudrate=baud, timeout=0.02, write_timeout=0.02)
        self.protocol = protocol
        self.yaw_kp = yaw_kp
        self.pitch_kp = pitch_kp
        self.yaw_kd = yaw_kd
        self.pitch_kd = pitch_kd
        self.max_yaw = max_yaw
        self.max_pitch = max_pitch
        self.deadzone = deadzone
        self.yaw_sign = yaw_sign
        self.pitch_sign = pitch_sign
        self.control_mode = control_mode
        self.max_yaw_step = max_yaw_step
        self.max_pitch_step = max_pitch_step
        self.laser_on_target = laser_on_target
        self.laser_deadzone = laser_deadzone
        self.target_offset_x = target_offset_x
        self.target_offset_y = target_offset_y
        self.last_yaw = 0.0
        self.last_pitch = 0.0
        self.target_yaw_rad = 0.0
        self.target_pitch_rad = 0.0
        self.laser_enabled = False
        self.target_centered = False
        self.prev_time = time.time()
        self.prev_yaw_err = None
        self.prev_pitch_err = None
        self.last_yaw_err = 0.0
        self.last_pitch_err = 0.0
        self.last_yaw_d_err = 0.0
        self.last_pitch_d_err = 0.0
        self.enable()
        self.enable_stability()
        self.init_position_target()

    def send(self, cmd, yaw=0.0, pitch=0.0):
        self.ser.write(make_gimbal_packet(cmd, yaw, pitch))

    def send_text(self, command):
        self.ser.write((command + '\r\n').encode('ascii'))

    def enable(self):
        self.send_text('enable') if self.protocol == 'text' else self.send(CMD_ENABLE)

    def disable(self):
        self.send_text('disable') if self.protocol == 'text' else self.send(CMD_DISABLE)

    def enable_stability(self):
        self.send_text('enable_stability') if self.protocol == 'text' else self.send(CMD_STABILITY_ON)

    def send_speed(self, yaw=0.0, pitch=0.0):
        if self.protocol == 'text':
            self.send_text(f'ctrl low_speed {yaw:.3f} {pitch:.3f}')
        else:
            self.send(CMD_LOW_SPEED, yaw=yaw, pitch=pitch)

    def send_angle(self, yaw=0.0, pitch=0.0):
        if self.protocol == 'text':
            self.send_text(f'ctrl angle {yaw:.4f} {pitch:.4f}')
        else:
            self.send(CMD_ANGLE, yaw=yaw, pitch=pitch)

    def set_laser(self, enabled):
        if not self.laser_on_target:
            enabled = False
        if self.laser_enabled == enabled:
            return
        if self.protocol == 'text':
            self.send_text('enable_laser' if enabled else 'disable_laser')
        else:
            self.send(CMD_LASER_ON if enabled else CMD_LASER_OFF)
        self.laser_enabled = enabled

    def init_position_target(self):
        if self.control_mode != 'angle' or self.protocol == 'text':
            return
        feedback = read_feedback_from_serial(self.ser)
        if feedback is None:
            print('Warning: cannot read initial gimbal angle; angle tracking starts from 0,0 rad.')
            return
        self.target_yaw_rad = feedback['yaw_motor_rad']
        self.target_pitch_rad = feedback['pitch_motor_rad']

    def update(self, target, frame_w, frame_h):
        now = time.time()
        dt = now - self.prev_time
        self.prev_time = now

        if target is None:
            self.target_centered = False
            self.set_laser(False)
            self.prev_yaw_err = None
            self.prev_pitch_err = None
            self.last_yaw_err = 0.0
            self.last_pitch_err = 0.0
            self.last_yaw_d_err = 0.0
            self.last_pitch_d_err = 0.0
            self.stop()
            return 0.0, 0.0, None, None

        target_center_x = frame_w / 2.0 + self.target_offset_x
        target_center_y = frame_h / 2.0 + self.target_offset_y
        raw_err_x = target["x"] - target_center_x
        raw_err_y = target["y"] - target_center_y
        self.target_centered = (
            abs(raw_err_x) <= self.laser_deadzone and
            abs(raw_err_y) <= self.laser_deadzone
        )

        err_x = raw_err_x
        err_y = raw_err_y
        if abs(err_x) < self.deadzone:
            err_x = 0.0
        if abs(err_y) < self.deadzone:
            err_y = 0.0

        yaw_norm = err_x / (frame_w / 2.0)
        pitch_norm = err_y / (frame_h / 2.0)
        yaw_d_err = 0.0 if self.prev_yaw_err is None or dt <= 0 else (yaw_norm - self.prev_yaw_err) / dt
        pitch_d_err = 0.0 if self.prev_pitch_err is None or dt <= 0 else (pitch_norm - self.prev_pitch_err) / dt
        self.prev_yaw_err = yaw_norm
        self.prev_pitch_err = pitch_norm
        self.last_yaw_err = yaw_norm
        self.last_pitch_err = pitch_norm
        self.last_yaw_d_err = yaw_d_err
        self.last_pitch_d_err = pitch_d_err

        if self.control_mode == 'angle':
            yaw_step = self.yaw_sign * (self.yaw_kp * yaw_norm + self.yaw_kd * yaw_d_err)
            pitch_step = self.pitch_sign * (self.pitch_kp * pitch_norm + self.pitch_kd * pitch_d_err)
            yaw_step = float(np.clip(yaw_step, -self.max_yaw_step, self.max_yaw_step))
            pitch_step = float(np.clip(pitch_step, -self.max_pitch_step, self.max_pitch_step))
            self.target_yaw_rad += yaw_step
            self.target_pitch_rad += pitch_step
            self.send_angle(yaw=self.target_yaw_rad, pitch=self.target_pitch_rad)
            yaw = self.target_yaw_rad
            pitch = self.target_pitch_rad
        else:
            yaw = self.yaw_sign * (self.yaw_kp * yaw_norm + self.yaw_kd * yaw_d_err)
            pitch = self.pitch_sign * (self.pitch_kp * pitch_norm + self.pitch_kd * pitch_d_err)
            yaw = float(np.clip(yaw, -self.max_yaw, self.max_yaw))
            pitch = float(np.clip(pitch, -self.max_pitch, self.max_pitch))
            self.send_speed(yaw=yaw, pitch=pitch)

        self.set_laser(self.target_centered)
        self.last_yaw = yaw
        self.last_pitch = pitch
        return yaw, pitch, err_x, err_y

    def stop(self):
        if self.control_mode == 'angle':
            self.last_yaw = self.target_yaw_rad
            self.last_pitch = self.target_pitch_rad
            return
        if self.last_yaw != 0.0 or self.last_pitch != 0.0:
            self.send_speed(yaw=0.0, pitch=0.0)
        self.last_yaw = 0.0
        self.last_pitch = 0.0

    def close(self, disable=False):
        self.set_laser(False)
        self.stop()
        if disable:
            self.disable()
        self.ser.close()


def atomic_write(path, data):
    fd, tmp = tempfile.mkstemp(dir=os.path.dirname(path), prefix=".red_track_")
    try:
        os.write(fd, json.dumps(data, ensure_ascii=False).encode("utf-8"))
        os.fsync(fd)
    finally:
        os.close(fd)
    os.replace(tmp, path)


def is_video_device_path(source):
    return re.fullmatch(r'/dev/video\d+', str(source)) is not None


def parse_camera_source(source):
    if source == 'auto':
        return source
    if is_video_device_path(source):
        return source
    try:
        return int(source)
    except (TypeError, ValueError):
        raise ValueError('source must be auto, camera index, or /dev/videoN')


def configure_capture(cap, width=None, height=None, fourcc=None):
    if fourcc:
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc))
    if width:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    if height:
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)


def open_capture(source, width=None, height=None, fourcc=None):
    if source == 'auto':
        candidates = sorted(glob.glob('/dev/video*')) + list(range(10))
        seen = set()
        for candidate in candidates:
            if candidate in seen:
                continue
            seen.add(candidate)
            cap = cv2.VideoCapture(candidate, cv2.CAP_V4L2)
            configure_capture(cap, width, height, fourcc)
            if cap.isOpened():
                ret, _ = cap.read()
                if ret:
                    print(f'Auto selected camera source: {candidate}')
                    return cap, candidate
            cap.release()
        return None, source

    cap = cv2.VideoCapture(source, cv2.CAP_V4L2)
    configure_capture(cap, width, height, fourcc)
    return cap, source


def post_process(outputs):
    boxes_all, scores_all, classes_all = [], [], []
    for i, stride in enumerate(STRIDES):
        box_feat = outputs[i * 3]
        cls_feat = outputs[i * 3 + 1]
        cls = cls_feat.reshape(cls_feat.shape[1], -1).T
        cls_max = cls.max(axis=1)
        mask = cls_max >= OBJ_THRESH
        if not np.any(mask):
            continue

        box = box_feat.reshape(64, -1).T[mask]
        grid = GRIDS[stride][mask]
        scores = cls_max[mask]
        classes = np.argmax(cls[mask], axis=1)

        b = box.reshape(-1, 4, 16)
        b = b - b.max(axis=2, keepdims=True)
        b = np.exp(b)
        b = b / b.sum(axis=2, keepdims=True)
        b = (b * DFL_PROJ).sum(axis=2)

        x1 = (grid[:, 0] + 0.5 - b[:, 0]) * stride
        y1 = (grid[:, 1] + 0.5 - b[:, 1]) * stride
        x2 = (grid[:, 0] + 0.5 + b[:, 2]) * stride
        y2 = (grid[:, 1] + 0.5 + b[:, 3]) * stride

        boxes_all.append(np.stack([x1, y1, x2, y2], axis=-1))
        scores_all.append(scores)
        classes_all.append(classes)

    if not boxes_all:
        return None, None, None

    boxes = np.concatenate(boxes_all)
    scores = np.concatenate(scores_all)
    classes = np.concatenate(classes_all)
    wh = boxes[:, 2:4] - boxes[:, 0:2]
    rects = np.concatenate([boxes[:, 0:2], wh], axis=1).tolist()
    keep = cv2.dnn.NMSBoxes(rects, scores.tolist(), OBJ_THRESH, NMS_THRESH)
    if len(keep) == 0:
        return None, None, None
    keep = np.array(keep).flatten()
    return boxes[keep], classes[keep], scores[keep]


def letterbox(im, new_shape=(640, 640), color=(0, 0, 0)):
    shape = im.shape[:2]
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw = (new_shape[1] - new_unpad[0]) / 2
    dh = (new_shape[0] - new_unpad[1]) / 2
    if shape[::-1] != new_unpad:
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return im, r, (dw, dh)


def find_uav_target(frame, rknn):
    img, ratio, pad = letterbox(frame, new_shape=IMG_SIZE)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = np.expand_dims(img, axis=0)

    outputs = rknn.inference(inputs=[img])
    boxes, classes, scores = post_process(outputs)
    if boxes is None:
        return None, None, None, None, ratio, pad

    detections = []
    frame_h, frame_w = frame.shape[:2]
    for box, score, cl in zip(boxes, scores, classes):
        x1 = int((box[0] - pad[0]) / ratio)
        y1 = int((box[1] - pad[1]) / ratio)
        x2 = int((box[2] - pad[0]) / ratio)
        y2 = int((box[3] - pad[1]) / ratio)
        x1 = int(np.clip(x1, 0, frame_w - 1))
        y1 = int(np.clip(y1, 0, frame_h - 1))
        x2 = int(np.clip(x2, 0, frame_w - 1))
        y2 = int(np.clip(y2, 0, frame_h - 1))
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        detections.append({
            "x": round(cx, 1),
            "y": round(cy, 1),
            "score": round(float(score), 3),
            "bbox": (x1, y1, max(1, x2 - x1), max(1, y2 - y1)),
            "class": CLASSES[int(cl)],
        })

    target = max(detections, key=lambda item: item["score"]) if detections else None
    return target, boxes, classes, scores, ratio, pad


def draw_uav_detections(frame, boxes, classes, scores, ratio, pad):
    if boxes is None:
        return frame
    frame_h, frame_w = frame.shape[:2]
    for box, cl, score in zip(boxes, classes, scores):
        x1 = int((box[0] - pad[0]) / ratio)
        y1 = int((box[1] - pad[1]) / ratio)
        x2 = int((box[2] - pad[0]) / ratio)
        y2 = int((box[3] - pad[1]) / ratio)
        x1 = int(np.clip(x1, 0, frame_w - 1))
        y1 = int(np.clip(y1, 0, frame_h - 1))
        x2 = int(np.clip(x2, 0, frame_w - 1))
        y2 = int(np.clip(y2, 0, frame_h - 1))
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f'{CLASSES[int(cl)]} {float(score):.2f}', (x1, max(0, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    return frame


def warn_if_no_display():
    if not os.environ.get('DISPLAY') and not os.environ.get('WAYLAND_DISPLAY'):
        print('Warning: no DISPLAY/WAYLAND_DISPLAY found; OpenCV window may not appear.')


def main():
    parser = argparse.ArgumentParser(description='Track a red block and control QGimbal.')
    parser.add_argument('--source', default=CAMERA_SOURCE, help='auto, camera index, or /dev/videoN')
    parser.add_argument('--det-file', default=DET_FILE, help='json tracking output path')
    parser.add_argument('--display', action='store_true', help='show realtime preview window')
    parser.add_argument('--width', type=int, default=640, help='camera capture width')
    parser.add_argument('--height', type=int, default=480, help='camera capture height')
    parser.add_argument('--fourcc', default='MJPG', help='camera pixel format')
    parser.add_argument('--min-area', type=float, default=80.0, help='minimum red contour area in pixels')
    parser.add_argument('--track', action='store_true', help='control gimbal to track red block')
    parser.add_argument('--uart-port', default=GIMBAL_PORT, help='gimbal serial device')
    parser.add_argument('--uart-baud', type=int, default=115200, help='gimbal UART baudrate')
    parser.add_argument('--gimbal-protocol', choices=('uart', 'text'), default='uart')
    parser.add_argument('--gimbal-control', choices=('speed', 'angle'), default='speed')
    parser.add_argument('--yaw-kp', type=float, default=50.0)
    parser.add_argument('--pitch-kp', type=float, default=40.0)
    parser.add_argument('--yaw-kd', type=float, default=0.0)
    parser.add_argument('--pitch-kd', type=float, default=0.0)
    parser.add_argument('--max-yaw-speed', type=float, default=30.0)
    parser.add_argument('--max-pitch-speed', type=float, default=25.0)
    parser.add_argument('--max-yaw-step', type=float, default=0.03)
    parser.add_argument('--max-pitch-step', type=float, default=0.025)
    parser.add_argument('--deadzone', type=float, default=20.0)
    parser.add_argument('--laser-deadzone', type=float, default=18.0)
    parser.add_argument('--target-offset-x', type=float, default=10.0)
    parser.add_argument('--target-offset-y', type=float, default=0.0)
    parser.add_argument('--yaw-sign', type=float, default=1.0)
    parser.add_argument('--pitch-sign', type=float, default=1.0)
    parser.add_argument('--no-laser', action='store_true')
    parser.add_argument('--keep-gimbal-enabled-on-exit', action='store_true')
    args = parser.parse_args()

    rknn = RKNNLite()
    print('--> Load RKNN model')
    assert rknn.load_rknn(RKNN_MODEL) == 0, 'Load RKNN model failed!'
    print('--> Init runtime')
    assert rknn.init_runtime(core_mask=RKNNLite.NPU_CORE_0_1_2) == 0

    source = parse_camera_source(args.source)
    cap, source = open_capture(source, width=args.width, height=args.height, fourcc=args.fourcc)
    if cap is None or not cap.isOpened():
        raise RuntimeError(f'Cannot open camera source: {source}')

    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 25
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f'Source: {source} | {w}x{h}, {fps}fps')
    print(f'红色块结果输出: {args.det_file}')

    if args.display:
        warn_if_no_display()
        cv2.namedWindow('red-track-gimbal', cv2.WINDOW_NORMAL)

    gimbal = None
    if args.track:
        gimbal = GimbalController(
            args.uart_port, args.uart_baud, args.gimbal_protocol,
            args.yaw_kp, args.pitch_kp, args.yaw_kd, args.pitch_kd,
            args.max_yaw_speed, args.max_pitch_speed, args.deadzone,
            args.yaw_sign, args.pitch_sign, args.gimbal_control,
            args.max_yaw_step, args.max_pitch_step, not args.no_laser,
            args.laser_deadzone, args.target_offset_x, args.target_offset_y,
        )
        print(
            f'Gimbal tracking enabled on {args.uart_port} @ {args.uart_baud}, '
            f'control={args.gimbal_control}, laser={"off" if args.no_laser else "auto"}'
        )

    frame_count = 0
    start = time.time()
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_count += 1
            if w <= 0 or h <= 0:
                h, w = frame.shape[:2]

            target, boxes, classes, scores, ratio, pad = find_uav_target(frame, rknn)
            yaw_cmd = pitch_cmd = err_x = err_y = None
            laser_on = False
            if gimbal is not None:
                yaw_cmd, pitch_cmd, err_x, err_y = gimbal.update(target, w, h)
                laser_on = gimbal.laser_enabled

            state = {
                "detected": target is not None,
                "x": target["x"] if target else None,
                "y": target["y"] if target else None,
                "score": target["score"] if target else None,
                "err_x": round(err_x, 1) if err_x is not None else None,
                "err_y": round(err_y, 1) if err_y is not None else None,
                "yaw_err_norm": round(gimbal.last_yaw_err, 4) if gimbal is not None else None,
                "pitch_err_norm": round(gimbal.last_pitch_err, 4) if gimbal is not None else None,
                "yaw_d_err": round(gimbal.last_yaw_d_err, 4) if gimbal is not None else None,
                "pitch_d_err": round(gimbal.last_pitch_d_err, 4) if gimbal is not None else None,
                "gimbal_control": args.gimbal_control if gimbal is not None else None,
                "target_centered": bool(gimbal.target_centered) if gimbal is not None else False,
                "laser": laser_on,
                "yaw": round(yaw_cmd, 4) if yaw_cmd is not None else None,
                "pitch": round(pitch_cmd, 4) if pitch_cmd is not None else None,
                "frame": frame_count,
            }
            atomic_write(args.det_file, state)

            frame = draw_uav_detections(frame, boxes, classes, scores, ratio, pad)
            if target:
                x, y, bw, bh = target["bbox"]
                cv2.rectangle(frame, (x, y), (x + bw, y + bh), (0, 255, 0), 2)
                cv2.circle(frame, (int(target["x"]), int(target["y"])), 4, (255, 0, 0), -1)
            cv2.circle(frame, (int(w / 2 + args.target_offset_x), int(h / 2 + args.target_offset_y)),
                       int(args.laser_deadzone), (0, 255, 255), 1)

            if args.display:
                cv2.imshow('red-track-gimbal', frame)
                if cv2.waitKey(1) & 0xFF in (27, ord('q')):
                    break

            if target:
                result_text = f'UAV: yes | x={target["x"]} y={target["y"]} score={target["score"]}'
            else:
                result_text = 'UAV: no | x=None y=None'
            if gimbal is not None:
                unit = 'rad' if args.gimbal_control == 'angle' else 'rpm'
                err_text = f'({err_x:.1f},{err_y:.1f})' if err_x is not None and err_y is not None else '(None,None)'
                result_text += (
                    f' | err={err_text}px'
                    f' | e=({gimbal.last_yaw_err:.3f},{gimbal.last_pitch_err:.3f})'
                    f' | de=({gimbal.last_yaw_d_err:.3f},{gimbal.last_pitch_d_err:.3f})/s'
                    f' | yaw={yaw_cmd:.3f}{unit} pitch={pitch_cmd:.3f}{unit}'
                    f' | centered={gimbal.target_centered} laser={laser_on}'
                )
            print(f'Frame {frame_count} | {result_text}', end='\r')
    except KeyboardInterrupt:
        print('\nInterrupted by Ctrl+C.')
    finally:
        if gimbal is not None:
            gimbal.close(disable=not args.keep_gimbal_enabled_on_exit)
        cap.release()
        rknn.release()
        if args.display:
            cv2.destroyAllWindows()

    elapsed = time.time() - start
    avg_fps = frame_count / elapsed if elapsed > 0 else 0.0
    print(f'\nDone! {frame_count} frames, Avg FPS: {avg_fps:.1f}')


if __name__ == '__main__':
    main()

