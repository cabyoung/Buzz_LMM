#!/usr/bin/env python3

import os
import time
import struct
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
import subprocess
from datetime import datetime, timedelta
import transforms3d as tf3d
from nav_msgs.msg import Odometry
import math
import argparse
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend (no display needed on robot)
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Command Line Arguments
def parse_arguments():
    parser = argparse.ArgumentParser(description='Buzz cam with configurable path points')
    parser.add_argument('--lookahead-dist', type=float, default=2.0,
                       help='Distance in meters to look ahead from current position (default: 2.0m, waypoints are ~0.1m apart)')
    parser.add_argument('--speed', type=float, default=0.25,
                       help='Estimated robot speed in m/s used for time labeling only (default: 0.25)')
    parser.add_argument('--interval', type=float, default=1.0,
                       help='Minimum seconds between path processing updates (default: 1.0, use 0 for every message)')
    parser.add_argument('--raw', action='store_true',
                       help='Write raw (x, y, yaw) data to Path.txt instead of processed commands')
    parser.add_argument('--visualize', action='store_true',
                       help='Save path visualizer PNGs (raw points with yaw arrows + processed command segments)')
    parser.add_argument('--slight-threshold', type=float, default=20.0,
                       help='Angle (degrees) between near/far heading below which path is "forward"; above is "slight turn" (default: 20)')
    parser.add_argument('--turn-threshold', type=float, default=45.0,
                       help='Angle (degrees) between near/far heading at or above which path is "turn" instead of "slight turn" (default: 45)')
    return parser.parse_args()

ARGS = parse_arguments()

# Global Variables
BASE_OUTPUT_DIR = "/home/unitree/buzz_out"
LOOKAHEAD_DIST = ARGS.lookahead_dist
ROBOT_SPEED = ARGS.speed
RAW_MODE = ARGS.raw
VISUALIZE = ARGS.visualize
PROCESSING_INTERVAL = ARGS.interval
SLIGHT_THRESHOLD = ARGS.slight_threshold
TURN_THRESHOLD = ARGS.turn_threshold
ITERATION = 0
LOCAL_IP = "192.168.123.222"
LOCAL_USER = "halab2"
REMOTE_BASE = "/home/halab2/buzz_shared"

def transfer_to_local(file_path, file_type="general"):
    """Transfer any file to local machine using SCP"""
    try:
        # Map file types to remote directories
        dir_map = {
            "path": "path_files",
        }
        remote_dir = dir_map.get(file_type, "")
        remote_path = f"{LOCAL_USER}@{LOCAL_IP}:{REMOTE_BASE}/{remote_dir}/"
        
        # Transfer the file
        result = subprocess.run(
            ['scp', file_path, remote_path],
            capture_output=True,
            text=True,
            timeout=30
        )
        
        if result.returncode == 0:
            print(f"Transferred {file_type}: {os.path.basename(file_path)}")
            return True
        else:
            print(f"Transfer failed: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        print(f"Transfer timeout for {file_path}")
        return False
    except Exception as e:
        print(f"Transfer error: {e}")
        return False


class Global_Path(Node):
    def __init__(self):
        super().__init__('Global_Path')
        self.pose_subscription = self.create_subscription(
            Odometry,
            '/uslam/localization/odom',
            self.pose_callback,
            10
        )

        self.path_subscription = self.create_subscription(
            PointCloud2, 
            '/uslam/navigation/global_path', 
            self.path_callback, 
            10
        )
        ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        self.last_processing_time = 0
        self.processing_interval = PROCESSING_INTERVAL
        self.latest_msg = None
        self.message_count = 0
        self.og_yaw = None
        self.output_filename = os.path.join(BASE_OUTPUT_DIR, "Path.txt")
        os.makedirs(BASE_OUTPUT_DIR, exist_ok=True)
        self.all_paths_file = os.path.join(BASE_OUTPUT_DIR, f"All_Paths_{ts}.txt")
        self.raw_path_file = os.path.join(BASE_OUTPUT_DIR, f"Raw_Path_{ts}.txt")
        # PNG filenames stamped at startup — overwritten each iteration to show full accumulated path
        self.raw_png = os.path.join(BASE_OUTPUT_DIR, f"viz_raw_{ts}.png")
        self.proc_png = os.path.join(BASE_OUTPUT_DIR, f"viz_proc_{ts}.png")
        # Accumulated data across all iterations for the full path visualization
        self.viz_data_list = []    # all raw (x, y, yaw) points
        self.viz_smooth_data = []  # all smoothed points
        self.viz_movements = []    # all commands
        self.get_logger().info(f"Waiting for Path...{self.processing_interval}")

    def write_to_file(self, text):
        """Write text to both console and file"""
        with open(self.output_filename, "w") as f:
            f.write(text)
            f.flush()
    
    def append_path(self, filename, text):
        """Write text to both console and file"""
        with open(filename, "a") as f:
            f.write(text)
            f.flush()

    def pose_callback(self, msg):
        o = msg.pose.pose.orientation
        quaternion = [o.w, o.x, o.y, o.z]
        euler = tf3d.euler.quat2euler(quaternion, 'szyx')
        yaw = euler[0]
        self.og_yaw = yaw
        #print(f"Original Yaw: {self.og_yaw:.2f}")
        #print("=============================\n")


    def path_callback(self, msg):
        current_time = time.time()
        self.latest_msg = msg
        self.message_count += 1
        output_lines = []
        print(f"\n=== Global Navigation Path Message #{self.message_count} ===")
        print(f"Timestamp: {self.rostime2strf(msg.header.stamp.sec, msg.header.stamp.nanosec)}")
        print(f"Frame ID: {msg.header.frame_id}")
        print(f"Path points: {msg.width}")
        if self.message_count == 1:
            print(f"Point fields: {[(f.name, f.datatype, f.offset) for f in msg.fields]}")
            print(f"Point step: {msg.point_step} bytes")
        print(f"Original Yaw: {self.og_yaw:.2f}" if self.og_yaw is not None else "Original Yaw: (waiting for pose)")
        if self.og_yaw is None:
            print("Skipping path processing: no pose received yet")
            return
        if current_time - self.last_processing_time >= self.processing_interval:
            self.last_processing_time = current_time
            # manual_decode_points handles printing + file writing; don't call print_path_coordinates too
            coordinates_output = self.manual_decode_points(msg)
            output_lines.extend(coordinates_output)
            output_lines.append("=" * 50 + "\n")
    
    def normalize_angle(self, angle):
        while angle > math.pi:
            angle -= 2 * math.pi
        while angle < -math.pi:
            angle += 2 * math.pi
        return angle

    def print_path_coordinates(self, msg):
        """Extract and print the x,y,z coordinates from PointCloud2 data"""
        if msg.width == 0:
            print("No points in path")
            return []
        
        print("\nPath Coordinates (x, y):")
        print("-" * 30)
        
        return self.manual_decode_points(msg)

    def manual_decode_points(self, msg):
        output_lines = []
        data_list = []
        global ITERATION
        ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")

        cumulative_dist = 0.0
        for i in range(msg.width):
            offset = i * msg.point_step
            x = struct.unpack('f', msg.data[offset:offset+4])[0]
            y = struct.unpack('f', msg.data[offset+4:offset+8])[0]

            if i > 0:
                prev = data_list[-1]
                dist = math.sqrt((x - prev[0])**2 + (y - prev[1])**2)
                cumulative_dist += dist
                if cumulative_dist > LOOKAHEAD_DIST:
                    break

            line = f"Point {i:2d}: ({x:7.3f}, {y:7.3f})  d={cumulative_dist:.2f}m"
            print(line)
            data_list.append([x, y, self.og_yaw])
            output_lines.append(line + "\n")
        
        # Compute TOTAL remaining path distance (beyond lookahead) for goal detection
        total_remaining = 0.0
        if msg.width > 1:
            for i in range(1, msg.width):
                o_prev = (i - 1) * msg.point_step
                o_curr = i * msg.point_step
                px = struct.unpack('f', msg.data[o_prev:o_prev+4])[0]
                py = struct.unpack('f', msg.data[o_prev+4:o_prev+8])[0]
                cx = struct.unpack('f', msg.data[o_curr:o_curr+4])[0]
                cy = struct.unpack('f', msg.data[o_curr+4:o_curr+8])[0]
                total_remaining += math.sqrt((cx - px)**2 + (cy - py)**2)

        # Detect goal states using TOTAL remaining distance, not just lookahead window
        PATH_DONE = (msg.width == 0) or (total_remaining < 2.0 and msg.width > 0)
        PATH_APPROACHING = not PATH_DONE and total_remaining < 4.0
        print(f"[GOAL] total_remaining={total_remaining:.2f}m  PATH_DONE={PATH_DONE}  PATH_APPROACHING={PATH_APPROACHING}")

        path, smooth_data = self.translate_nl(data_list, msg, speed=ROBOT_SPEED)
        rle = self.rle_path(path)
        full_rle = self.full_rle_path(msg)
        print(f"PATH: {rle}  |  FULL PATH: {full_rle}  |  PATH_DONE: {PATH_DONE}")
        try:
            if PATH_DONE:
                content = f"USLAM Global Path Log - Started at {ts}, Iteration {ITERATION}\n\n PATH: {rle}\nFULL PATH: {full_rle}\nPATH DONE"
                print("[GOAL] Writing PATH DONE to file")
            elif PATH_APPROACHING:
                content = f"USLAM Global Path Log - Started at {ts}, Iteration {ITERATION}\n\n PATH: {rle}\nFULL PATH: {full_rle}\nPATH APPROACHING"
                print(f"[GOAL] Writing PATH APPROACHING to file ({total_remaining:.2f}m remaining)")
            elif RAW_MODE:
                content = f"USLAM Global Path Log - Started at {ts}, Iteration {ITERATION}\n\n PATH: {data_list}\nFULL PATH: {full_rle}"
                print(f"[RAW MODE] Writing raw (x, y, yaw) to Path.txt")
            else:
                content = f"USLAM Global Path Log - Started at {ts}, Iteration {ITERATION}\n\n PATH: {rle}\nFULL PATH: {full_rle}"
            self.write_to_file(content)
            self.append_path(self.all_paths_file, f"USLAM Global Path Log - Started at {ts}, Iteration {ITERATION}\nPATH: {rle}  PATH_DONE: {PATH_DONE}\n\n")
            self.append_path(self.raw_path_file, f"USLAM Global Path Log - Started at {ts}, Iteration {ITERATION}\nPATH: {data_list}\n\n")

        except Exception:
            self.write_to_file(f"Path is empty \n data: {data_list}")

        if VISUALIZE and data_list:
            self.viz_data_list.extend(data_list)
            self.viz_smooth_data.extend(smooth_data)
            self.viz_movements.extend(path)
            self.save_visualizations(self.viz_data_list, self.viz_smooth_data, self.viz_movements, ts)

        if self.output_filename and os.path.exists(self.output_filename):
            os.sync()
            time.sleep(0.1)
            transfer_to_local(self.output_filename, "path")
            transfer_to_local(self.all_paths_file, "path")
            transfer_to_local(self.raw_path_file, "path")
            if VISUALIZE and data_list:
                transfer_to_local(self.raw_png, "path")
                transfer_to_local(self.proc_png, "path")
            ITERATION += 1
        return output_lines

    def full_rle_path(self, msg):
        """RLE-encode the ENTIRE remaining path (all waypoints, no lookahead cap).
        Handles multiple sequential turns by updating the reference heading after each step.
        Returns a human-readable string ending with 'then goal'."""
        if msg.width < 2:
            return "forward then goal"

        # Extract all waypoints
        full_data = []
        for i in range(msg.width):
            offset = i * msg.point_step
            x = struct.unpack('f', msg.data[offset:offset+4])[0]
            y = struct.unpack('f', msg.data[offset+4:offset+8])[0]
            full_data.append((x, y))

        current_yaw = self.og_yaw
        movements = []
        for i in range(1, len(full_data)):
            dx = full_data[i][0] - full_data[i-1][0]
            dy = full_data[i][1] - full_data[i-1][1]
            dist = math.sqrt(dx**2 + dy**2)
            if dist < 0.001:
                continue
            segment_yaw = math.atan2(dy, dx)
            turn_deg = math.degrees(self.normalize_angle(segment_yaw - current_yaw))
            abs_deg = abs(turn_deg)
            if abs_deg < SLIGHT_THRESHOLD:
                cmd = "forward"
            elif abs_deg < TURN_THRESHOLD:
                cmd = "forward"  # slight turns treated as forward
            else:
                cmd = "turn left" if turn_deg > 0 else "turn right"
            movements.append((cmd, dist))
            current_yaw = segment_yaw  # update heading for next step

        if not movements:
            return "forward then goal"

        # RLE compress
        segments = []
        prev_cmd, seg_dist = movements[0]
        for cmd, dist in movements[1:]:
            if cmd == prev_cmd:
                seg_dist += dist
            else:
                segments.append(f"{prev_cmd} {seg_dist:.1f}m")
                prev_cmd, seg_dist = cmd, dist
        segments.append(f"{prev_cmd} {seg_dist:.1f}m")

        total_dist = sum(d for _, d in movements)
        total_time = total_dist / ROBOT_SPEED
        return f"(total {total_dist:.1f}m, ~{total_time:.0f}s) {' then '.join(segments)} then goal"

    def rle_path(self, movements):
        """Run-length encode movements into a distance-labeled string for Path.txt."""
        import re
        entries = []
        for m in movements:
            cmd_m = re.search(r"command: ([^,]+)", m)
            dist_m = re.search(r"dist: ([0-9.]+)m", m)
            if cmd_m and dist_m:
                cmd = cmd_m.group(1)
                if cmd in ("slight turn left", "slight turn right"):
                    cmd = "forward"
                entries.append((cmd, float(dist_m.group(1))))
        if not entries:
            return "forward (no turns ahead)"
        total_dist = sum(d for _, d in entries)
        total_time = total_dist / ROBOT_SPEED
        segments = []
        prev_cmd, seg_dist = entries[0]
        for cmd, dist in entries[1:]:
            if cmd == prev_cmd:
                seg_dist += dist
            else:
                segments.append(f"{prev_cmd} {seg_dist:.1f}m")
                prev_cmd, seg_dist = cmd, dist
        segments.append(f"{prev_cmd} {seg_dist:.1f}m")
        return f"(next {total_dist:.1f}m, ~{total_time:.0f}s) {' then '.join(segments)}"

    def step(self, prev, current, current_yaw):
        dx = current[0] - prev[0]
        dy = current[1] - prev[1]
        print(f"DX:{dx}, DY:{dy}")
        if dx == 0 and dy == 0:
            return "forward"
        segment_yaw = math.atan2(dy, dx)
        turn_angle_rad = self.normalize_angle(segment_yaw - current_yaw)
        turn_angle_deg = math.degrees(turn_angle_rad)
        abs_deg = abs(turn_angle_deg)
        if abs_deg < SLIGHT_THRESHOLD:
            return "forward"
        elif abs_deg < TURN_THRESHOLD:
            return "slight turn left" if turn_angle_deg > 0 else "slight turn right"
        else:
            return "turn left" if turn_angle_deg > 0 else "turn right"

    def rostime2strf(self, ros_sec, ros_nano):
        sec = ros_sec + (ros_nano * 1e-9)
        dt = datetime.fromtimestamp(sec)
        return dt.strftime('%Y%m%d_%H%M%S_%f')
        
    def overall_turn(self, data_list, current_yaw):
        """Compare heading of first vs last segment for robust turn detection.
        Uses look-ahead direction comparison rather than step-by-step, which is
        immune to the angle-reduction caused by moving-average smoothing."""
        n = len(data_list)
        seg = min(5, n // 3)  # points per comparison window

        near_dx = data_list[seg][0] - data_list[0][0]
        near_dy = data_list[seg][1] - data_list[0][1]
        far_dx  = data_list[-1][0]  - data_list[-1 - seg][0]
        far_dy  = data_list[-1][1]  - data_list[-1 - seg][1]

        if abs(near_dx) < 0.001 and abs(near_dy) < 0.001:
            return "forward"

        near_yaw = math.atan2(near_dy, near_dx)
        far_yaw  = math.atan2(far_dy, far_dx) if (abs(far_dx) > 0.001 or abs(far_dy) > 0.001) else near_yaw

        turn_deg = math.degrees(self.normalize_angle(far_yaw - near_yaw))
        abs_deg  = abs(turn_deg)

        if abs_deg < SLIGHT_THRESHOLD:
            return "forward"
        elif abs_deg < TURN_THRESHOLD:
            return "slight turn left" if turn_deg > 0 else "slight turn right"
        else:
            return "turn left" if turn_deg > 0 else "turn right"

    def translate_nl(self, data_list, msg, window_size=3, speed=0.25):
        movements = []
        ts = self.rostime2strf(msg.header.stamp.sec, msg.header.stamp.nanosec)
        ts_converted = datetime.fromtimestamp(msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9)

        if len(data_list) < 6:
            movements.append(f"Path not long enough")
            return [], []

        # Smooth data for visualization only (not used for classification)
        smooth_data = []
        for i in range(len(data_list)):
            start = max(0, i - window_size//2)
            end = min(len(data_list), i + window_size//2 + 1)
            window = data_list[start:end]
            avg_x = np.mean([p[0] for p in window])
            avg_y = np.mean([p[1] for p in window])
            smooth_data.append((avg_x, avg_y, 0))

        current_yaw = self.og_yaw

        # Look-ahead: determine overall direction using first vs last segment heading
        overall_direction = self.overall_turn(data_list, current_yaw)

        # Find where the turn begins by scanning raw points for first significant deviation
        steps_forward = len(data_list) - 1  # default: all forward
        if overall_direction != "forward":
            steps_forward = 0
            for i in range(1, len(data_list)):
                dx = data_list[i][0] - data_list[i - 1][0]
                dy = data_list[i][1] - data_list[i - 1][1]
                if abs(dx) < 0.001 and abs(dy) < 0.001:
                    steps_forward += 1
                    continue
                seg_yaw  = math.atan2(dy, dx)
                seg_turn = abs(math.degrees(self.normalize_angle(seg_yaw - current_yaw)))
                if seg_turn < SLIGHT_THRESHOLD:
                    steps_forward += 1
                else:
                    break

        # Build movement list with timestamps
        for i in range(1, len(data_list)):
            cmd = "forward" if (i - 1) < steps_forward else overall_direction
            dx = data_list[i][0] - data_list[i - 1][0]
            dy = data_list[i][1] - data_list[i - 1][1]
            distance = np.sqrt(dx**2 + dy**2)
            ts_converted += timedelta(seconds=distance / speed)
            ts_str = ts_converted.strftime('%Y%m%d_%H%M%S_%f')
            movements.append(f"command: {cmd}, dist: {distance:.3f}m, time: {ts_str}")

        return movements, smooth_data
    
    def save_visualizations(self, data_list, smooth_data, movements, ts):
        """Save accumulated path visualizer PNGs (overwritten each iteration)"""
        raw_png  = self.raw_png
        proc_png = self.proc_png

        xs = [p[0] for p in data_list]
        ys = [p[1] for p in data_list]
        yaws = [p[2] for p in data_list]

        # --- Raw plot: points + yaw arrows ---
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.plot(xs, ys, 'b-', linewidth=1.5, alpha=0.5, label='Path')
        ax.scatter(xs, ys, c='blue', s=20, zorder=3)
        ax.scatter(xs[0], ys[0], c='green', s=80, zorder=4, label='Start')
        ax.scatter(xs[-1], ys[-1], c='red', s=80, zorder=4, label='End')
        arrow_len = (max(max(xs)-min(xs), max(ys)-min(ys), 0.1)) * 0.06
        for x, y, yaw in zip(xs, ys, yaws):
            if yaw is not None:
                ax.annotate('', xy=(x + arrow_len*math.cos(yaw), y + arrow_len*math.sin(yaw)),
                            xytext=(x, y),
                            arrowprops=dict(arrowstyle='->', color='orange', lw=1.2))
        ax.set_title(f'Raw Path (x, y, yaw) — updated {ts}\nTotal points: {len(data_list)} (iter {ITERATION})')
        ax.set_xlabel('X (m)'); ax.set_ylabel('Y (m)')
        ax.legend(); ax.set_aspect('equal'); ax.grid(True)
        fig.savefig(raw_png, dpi=120, bbox_inches='tight')
        plt.close(fig)
        print(f"[VIZ] Saved raw PNG: {raw_png}")

        # --- Processed plot: color-coded command segments ---
        cmd_colors = {'forward': 'green', 'slight turn left': 'cyan', 'slight turn right': 'orange',
                      'turn left': 'blue', 'turn right': 'red', 'None': 'gray'}
        fig, ax = plt.subplots(figsize=(8, 8))
        if smooth_data and len(smooth_data) > 1:
            sxs = [p[0] for p in smooth_data]
            sys_ = [p[1] for p in smooth_data]
            ax.plot(sxs, sys_, 'k--', linewidth=0.8, alpha=0.3, label='Smoothed path')
            for i, cmd_str in enumerate(movements):
                if i + 1 >= len(smooth_data):
                    break
                # Parse command out of "command: forward, time: ..."
                cmd = 'None'
                if 'command: ' in cmd_str:
                    cmd = cmd_str.split('command: ')[1].split(',')[0].strip()
                color = cmd_colors.get(cmd, 'gray')
                ax.plot([smooth_data[i][0], smooth_data[i+1][0]],
                        [smooth_data[i][1], smooth_data[i+1][1]],
                        color=color, linewidth=2.5)
            ax.scatter(sxs[0], sys_[0], c='green', s=80, zorder=4)
            ax.scatter(sxs[-1], sys_[-1], c='red', s=80, zorder=4)
        else:
            ax.text(0.5, 0.5, 'Not enough data', transform=ax.transAxes, ha='center')
        legend_patches = [mpatches.Patch(color=c, label=l) for l, c in cmd_colors.items()]
        legend_patches += [mpatches.Patch(color='green', label='Start'),
                           mpatches.Patch(color='red', label='End')]
        ax.set_title(f'Processed Commands — updated {ts}\nTotal commands: {len(movements)} (iter {ITERATION})')
        ax.set_xlabel('X (m)'); ax.set_ylabel('Y (m)')
        ax.legend(handles=legend_patches); ax.set_aspect('equal'); ax.grid(True)
        fig.savefig(proc_png, dpi=120, bbox_inches='tight')
        plt.close(fig)
        print(f"[VIZ] Saved processed PNG: {proc_png}")

    def __del__(self):
        if hasattr(self, 'output_file') and self.output_file:
            self.output_file.close()

def main():
    rclpy.init()
    node = Global_Path()
    
    try:
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.1)
    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()
