#!/usr/bin/env python3
"""Benchmark motion detection CPU load at different resolutions.

Run this directly on your RK3566 to measure real-world performance.

Usage:
    python tools/benchmark_motion.py                         # synthetic frames
    python tools/benchmark_motion.py --camera 0              # live camera
    python tools/benchmark_motion.py --width 1280 --height 720
    python tools/benchmark_motion.py --width 640 --height 480
"""
import os
import sys
import time
import argparse
import statistics

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from core.motion import MotionDetector

def generate_test_frame(w, h, pattern='noise'):
    """Generate a synthetic frame with controllable motion."""
    if pattern == 'noise':
        # Random noise — worst case for motion detection
        return np.random.randint(0, 256, (h, w, 3), dtype=np.uint8)
    elif pattern == 'solid':
        return np.full((h, w, 3), 128, dtype=np.uint8)
    elif pattern == 'gradient':
        x = np.tile(np.arange(w, dtype=np.uint8), (h, 1))
        return cv2.merge([x, x, x])
    elif pattern == 'moving_box':
        frame = np.zeros((h, w, 3), dtype=np.uint8)
        offset = (int(time.time() * 10) % (w - 100))
        cv2.rectangle(frame, (offset, 50), (offset + 100, 150), (255, 255, 255), -1)
        return frame
    else:
        return np.random.randint(0, 256, (h, w, 3), dtype=np.uint8)


def benchmark_synthetic(widths=[(640, 480), (1280, 720)], iterations=200):
    """Benchmark using synthetic frames."""
    results = {}
    for w, h in widths:
        md = MotionDetector(threshold=25, min_area=500, blur_size=21)
        frame1 = generate_test_frame(w, h, 'noise')
        frame2 = generate_test_frame(w, h, 'noise')  # different noise = motion

        times = []
        # Warmup
        for _ in range(5):
            md.detect(frame1)

        # Benchmark
        for i in range(iterations):
            t0 = time.perf_counter()
            md.detect(frame2 if i % 2 == 0 else frame1)
            elapsed = time.perf_counter() - t0
            times.append(elapsed * 1000)  # ms

        results[(w, h)] = {
            'times': times,
            'min_ms': min(times),
            'max_ms': max(times),
            'avg_ms': statistics.mean(times),
            'median_ms': statistics.median(times),
            'p99_ms': sorted(times)[int(len(times) * 0.99)],
        }
    return results


def benchmark_camera(camera_index, width, height, warmup=30, iterations=150):
    """Benchmark using live camera feed."""
    cap = cv2.VideoCapture(camera_index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, 30)
    if not cap.isOpened():
        print(f"ERROR: Cannot open camera {camera_index}")
        sys.exit(1)

    # Check actual resolution
    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"  Camera actual resolution: {actual_w}x{actual_h}")

    md = MotionDetector(threshold=25, min_area=500, blur_size=21)
    times = []

    print(f"  Warming up ({warmup} frames)...")
    for i in range(warmup):
        ret, frame = cap.read()
        if not ret:
            continue
        md.detect(frame)

    print(f"  Benchmarking ({iterations} frames)...")
    for i in range(iterations):
        ret, frame = cap.read()
        if not ret:
            continue
        t0 = time.perf_counter()
        md.detect(frame)
        elapsed = time.perf_counter() - t0
        times.append(elapsed * 1000)

    cap.release()
    return {
        'times': times,
        'min_ms': min(times),
        'max_ms': max(times),
        'avg_ms': statistics.mean(times),
        'median_ms': statistics.median(times),
        'p99_ms': sorted(times)[int(len(times) * 0.99)],
    }


def print_header():
    print("=" * 72)
    print("  运动检测 CPU 性能基准测试 (Motion Detection Benchmark)")
    print("=" * 72)

    import platform
    uname = os.uname() if hasattr(os, 'uname') else None
    if uname:
        print(f"  平台: {uname.sysname} {uname.machine}")
        try:
            with open('/proc/cpuinfo') as f:
                for line in f:
                    if 'model name' in line or 'Hardware' in line or 'CPU part' in line:
                        print(f"  {line.strip()}")
        except:
            pass

    # Check OpenCV build info
    print(f"  OpenCV: {cv2.__version__}")
    build_info = cv2.getBuildInformation()
    if 'NEON' in build_info:
        print("  NEON:   ✅ 已启用")
    else:
        print("  NEON:   ❌ 未启用")
    print()


def print_results(results, label=""):
    if label:
        print(f"  [{label}]")
    print(f"  {'Resolution':>12} | {'Min(ms)':>8} | {'Avg(ms)':>8} | {'Median(ms)':>10} | {'Max(ms)':>8} | {'P99(ms)':>8}")
    print(f"  {'-'*12}-+-{'-'*8}-+-{'-'*8}-+-{'-'*10}-+-{'-'*8}-+-{'-'*8}")

    for (w, h), r in sorted(results.items()):
        print(f"  {w:>4}×{h:<5} | {r['min_ms']:>7.2f} | {r['avg_ms']:>7.2f} | {r['median_ms']:>9.2f} | {r['max_ms']:>7.2f} | {r['p99_ms']:>7.2f}")
    print()


def estimate_pipeline(resolution, motion_ms, fps=15):
    """Estimate pipeline budget at given FPS."""
    pipeline_ms = 1000 / fps
    remaining = pipeline_ms - motion_ms
    w, h = resolution
    pixels = w * h

    print(f"\n  ─── 流水线预算估算 ({w}×{h} @ {fps} FPS) ───")
    print(f"  每帧预算:          {pipeline_ms:.1f} ms")
    print(f"  运动检测:          {motion_ms:.1f} ms ({motion_ms/pipeline_ms*100:.0f}%)")
    print(f"  剩余预算:          {remaining:.1f} ms")
    print(f"  帧拷贝 (copy):     ~{pixels*3/1024/1024*0.5:.2f} ms")
    print(f"  录像写入 (write):  ~{pixels/307200*3:.1f} ms @ 640×480 基准")
    print(f"  ───")
    if remaining < 10:
        print(f"  ⚠️  剩余预算较少，高负载时可能出现帧率下降")
    else:
        print(f"  ✅ 预算充裕")
    print()


def main():
    parser = argparse.ArgumentParser(description='Motion detection benchmark')
    parser.add_argument('--camera', type=int, default=None,
                        help='Camera index for live benchmark')
    parser.add_argument('--width', type=int, default=None,
                        help='Single resolution width (omit for both)')
    parser.add_argument('--height', type=int, default=None,
                        help='Single resolution height')
    parser.add_argument('--iterations', type=int, default=200,
                        help='Number of iterations (default: 200)')
    parser.add_argument('--fps', type=int, default=15,
                        help='Target pipeline FPS (default: 15)')
    args = parser.parse_args()

    print_header()

    if args.camera is not None:
        # Live camera benchmark
        if args.width and args.height:
            sizes = [(args.width, args.height)]
        else:
            sizes = [(640, 480), (1280, 720)]

        all_results = {}
        for w, h in sizes:
            print(f"  ─── 摄像头 {args.camera} @ {w}×{h} ───")
            r = benchmark_camera(args.camera, w, h, iterations=args.iterations)
            all_results[(w, h)] = r
            print()
        print_results(all_results, label="摄像头模式")

        for (w, h), r in sorted(all_results.items()):
            estimate_pipeline((w, h), r['avg_ms'], fps=args.fps)
    else:
        # Synthetic benchmarks at both resolutions (or single)
        if args.width and args.height:
            widths = [(args.width, args.height)]
        else:
            widths = [(640, 480), (1280, 720), (1920, 1080)]

        print("  [合成帧模式 — 随机噪声（最坏情况）]")
        results = benchmark_synthetic(widths=widths, iterations=args.iterations)
        print_results(results, label="随机噪声")

        for (w, h), r in sorted(results.items()):
            estimate_pipeline((w, h), r['avg_ms'], fps=args.fps)

        print("  💡 提示: 随机噪声是 motion detection 的最坏情况（每个像素都变化）")
        print("           真实监控场景中，实际耗时通常比这个数字低 30-50%")
        print()


if __name__ == '__main__':
    main()
