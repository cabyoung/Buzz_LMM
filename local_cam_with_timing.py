#: realcam.py with timing and pickle integration for latency analysis
from turtle import stamp
from gtts import gTTS
from playsound import playsound
import pyrealsense2 as rs
import anthropic
import cv2
import numpy as np
from ultralytics import YOLO
import math
import time
import logging
import os
import threading
from collections import deque
from datetime import datetime
from datetime import timedelta
import srt
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node
from geometry_msgs.msg import Quaternion
import numpy as np
from sensor_msgs.msg import PointCloud2
import struct
import requests
import socket
import subprocess
from queue import Queue, Empty
import pickle

# ============================================================================
# TIMING AND PICKLE CONFIGURATION
# ============================================================================
TIMING_DATA = []  # Store all timing measurements
PICKLE_DIR = os.path.join("/home/halab2/buzz_shared", "timing_data")
os.makedirs(PICKLE_DIR, exist_ok=True)

class TimingContext:
    """Context manager for timing operations"""
    def __init__(self, operation_name, metadata=None):
        self.operation_name = operation_name
        self.metadata = metadata or {}
        self.start_time = None
        self.end_time = None
        
    def __enter__(self):
        self.start_time = time.perf_counter()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.perf_counter()
        duration = self.end_time - self.start_time
        
        timing_entry = {
            'operation': self.operation_name,
            'start_time': self.start_time,
            'end_time': self.end_time,
            'duration_ms': duration * 1000,
            'timestamp': datetime.now().isoformat(),
            'metadata': self.metadata
        }
        
        TIMING_DATA.append(timing_entry)
        print(f"⏱️  {self.operation_name}: {duration*1000:.2f}ms")
        
        return False  # Don't suppress exceptions

def save_timing_data():
    """Save timing data to pickle file"""
    if not TIMING_DATA:
        return
        
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    pickle_file = os.path.join(PICKLE_DIR, f"timing_data_{timestamp}.pkl")
    
    try:
        with open(pickle_file, 'wb') as f:
            pickle.dump(TIMING_DATA, f)
        print(f"✅ Saved timing data to {pickle_file}")
        
        # Also save a human-readable summary
        summary_file = os.path.join(PICKLE_DIR, f"timing_summary_{timestamp}.txt")
        with open(summary_file, 'w') as f:
            f.write("=== TIMING SUMMARY ===\n\n")
            
            # Group by operation type
            operations = {}
            for entry in TIMING_DATA:
                op = entry['operation']
                if op not in operations:
                    operations[op] = []
                operations[op].append(entry['duration_ms'])
            
            for op, durations in sorted(operations.items()):
                avg = sum(durations) / len(durations)
                min_val = min(durations)
                max_val = max(durations)
                f.write(f"{op}:\n")
                f.write(f"  Count: {len(durations)}\n")
                f.write(f"  Average: {avg:.2f}ms\n")
                f.write(f"  Min: {min_val:.2f}ms\n")
                f.write(f"  Max: {max_val:.2f}ms\n\n")
                
        print(f"✅ Saved timing summary to {summary_file}")
        
    except Exception as e:
        print(f"❌ Error saving timing data: {e}")

# ============================================================================
# ORIGINAL CODE WITH TIMING INTEGRATION
# ============================================================================

CLIENT = anthropic.Anthropic(api_key='REPLACE_WITH_YOUR_API_KEY')

caption_history = []
latest_caption = None
audio_playing = False
audio_control_lock = threading.Lock()

# --- Global variables for sound thread communication ---
sound_request_queue = deque()
sound_thread_event = threading.Event() # To signal the sound thread
sound_thread_active = True # Flag to control the sound worker thread's loop

# --- Webcam Control Flag (if you also want to stop the main loop from external signal) ---
camera_active = True # Added for consistent exit management

# --- Caption Text Locked Variable for threading ---
current_caption = ""
caption_lock = threading.Lock()  # Initialize global variable

# --- Global Variables for Minimized SRT files ---
last_labels = []  # Store past labels for logging
dist_threshold = 0.2  # Distance threshold for object detection to avoid duplicates
class_depth_history = []  # Store past depth values for each class


# --- Global Variables for API Queue ---
API_BUFF = deque(maxlen=3)
HIST_BUFF = []
FRAME_Q = Queue(maxsize=30)
processing_active = True

# --- Global Voice and Video
# filename array
VOICE = []
# video filename
VIDEO = ""
# --- Global SRT Files ---
BASE_OUTPUT_DIR = "/home/halab2/buzz_shared"
OUT_DIR = os.path.join(BASE_OUTPUT_DIR, "out_files")
IN_DIR = os.path.join(BASE_OUTPUT_DIR, "in_files")
AUDIO_DIR = os.path.join(BASE_OUTPUT_DIR, "audio_files")
VIDEO_DIR = os.path.join(BASE_OUTPUT_DIR, "video_files")
SCREENSHOT_DIR = os.path.join(BASE_OUTPUT_DIR, "screenshots")
PATH = []

out_filename = os.path.join(OUT_DIR, f"output_{datetime.now().strftime('%Y%m%d_%H%M%S')}.srt")
with open(out_filename, 'w', encoding='utf-8') as f:
        f.write("")
in_filename = os.path.join(IN_DIR, f"input_{datetime.now().strftime('%Y%m%d_%H%M%S')}.srt")
with open(in_filename, 'w', encoding='utf-8') as f:
        f.write("")


def check_path(filename="/home/halab2/buzz_shared/path_files/Path.txt"):
    global PATH
    with TimingContext("check_path"):
        if os.path.exists(filename):
            with open(filename, 'r') as f:
                lines = f.readlines()
                PATH.clear()
                for line in lines:
                    line = line.strip()
                    if line and not line.startswith("USLAM") and not line.startswith("=") and line != "PATH DONE":
                        PATH.append(line)
                print(f"Path loaded with {len(PATH)} points.")
        return PATH

def srt_file(filename, current_caption="NA", log="", start=0):
    global PATH
    with TimingContext("srt_file_write"):
        if not PATH:
            PATH = ["No path data available."]
        with open(filename, 'a', encoding='utf-8') as f:
            f.write(f"Timestamp: {start}\n")
            f.write(f"Caption: {current_caption}\n")
            f.write(f"Object detection log: {log}\n")
            f.write("\n")
            API_BUFF.append(f"Timestamp: {start} | Caption: {current_caption} | Object detection log: {log}") 

def srt_out(filename, current_caption="NA", log=""):
    global caption_history
    with TimingContext("srt_out_write"):
        with open(filename, 'a', encoding='utf-8') as f:
            f.write(f"Object detection log: {log}\n")
            f.write(f"Caption: {current_caption}\n")
            f.write(f"Claude API BUFF: {API_BUFF}\n")
            f.write(f"Caption History: {caption_history}\n")
            f.write("\n")

def video_init_realsense():
    with TimingContext("video_init"):
        # RealSense typically uses 640x480 resolution
        width = 640 * 2  # Double width for side-by-side display
        height = 480
        fps = 60 #15 or 30
        
        print(f"RealSense Camera: {640}x{480} @ {fps}fps")
        
        # Prepare video writer
        codecs = [('mp4v', '.mp4'), ('avc1', '.mp4'), ('XVID', '.avi')]
        writer = None
        for codec, ext in codecs:
            output_file = f"{'./'}realsense_record_{datetime.now().strftime('%Y%m%d_%H%M%S')}{ext}"
            VIDEO = output_file
            fourcc = cv2.VideoWriter_fourcc(*codec)
            writer = cv2.VideoWriter(output_file, fourcc, fps, (width, height))
            if writer.isOpened():
                print(f"Using codec: {codec}")
                break
            writer = None
        
        if not writer:
            print("ERROR: No working codec found")
            return None, None
        
        return writer, output_file

def claude(label):
    """Call Claude API with timing"""
    prompt = "You are given a number of seconds worth of data from three consecutive timestamps. Using this data, tell the human why you are moving a certain way and what you are avoiding. Also, mention any objects in the frames and their distance from the human. Let the person know if the given path must change due to obstacles." \
        " Keep the response concise and relevant to the human's navigation."
    system_prompt = "Responses need to be 8 words maximum and written in the point of view of a robot guide dog. The robot guide dog is helping a human to navigate around obstacles in the space. " \
    "Make sure your response helps the human understand why you are moving a certain way. Do not give specific information about the location of the objects. For example, if the depth data is 0.5m, do not say 'the object is 0.5m away'. Instead, say something like 'the object is close to you'. " \
    "Do not mind small changes in the FOV or depth data. Pay attention to and comment on larger changes in data among frame data given to you"
    
    with TimingContext("claude_api_call", metadata={'label_length': len(str(label))}):
        try:
            response = CLIENT.messages.create(
                model="claude-sonnet-4-5-20250929",
                max_tokens=100,
                system=system_prompt,
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": label},
                        {"type": "text", "text": prompt}
                    ]
                }]
            )

            response_content = response.content[0].text
            print(f"✓ Claude API response received: {response_content}")
            return response_content

        except anthropic.APIConnectionError as e:
            print(f"API Connection Error: {e}")
            return "API Connection Error"
        except anthropic.APIError as e:
            print(f"API Error: {e}")
            return "API Error"
        except Exception as e:
            print(f"Unexpected error: {e}")
            return "Error"

def frame_thread_worker():
    """Process frames from queue with timing"""
    global processing_active, current_caption
    
    while processing_active:
        try:
            with TimingContext("frame_queue_wait"):
                frame_data = FRAME_Q.get(timeout=1.0)
            
            timestamp = frame_data['timestamp']
            labels = frame_data['labels']
            combined = frame_data['combined']
            
            with TimingContext("frame_processing_total", metadata={'num_labels': len(labels)}):
                # Save frame
                with TimingContext("save_screenshot"):
                    frame_path = os.path.join(SCREENSHOT_DIR, f"frame_{timestamp}.jpg")
                    cv2.imwrite(frame_path, combined)
                
                # Log to SRT
                srt_file(in_filename, "NA", labels, timestamp)
                
                # Generate caption when buffer is full
                if len(API_BUFF) >= 3:
                    with TimingContext("caption_generation"):
                        caption = claude("\n".join(API_BUFF))
                        
                        with caption_lock:
                            current_caption = caption
                        
                        caption_history.append({
                            'timestamp': timestamp,
                            'caption': caption,
                            'labels': labels
                        })
                        
                        srt_out(out_filename, caption, "\n".join(API_BUFF))
                        API_BUFF.clear()
                        
        except Empty:
            continue
        except Exception as e:
            print(f"Error in frame processing thread: {e}")
            
    print("Frame processing thread exiting...")

def audio_thread_worker():
    """Handle audio generation and playback with timing"""
    global processing_active, audio_playing
    
    while processing_active:
        try:
            with caption_lock:
                caption = current_caption
                
            if caption and not audio_playing:
                with audio_control_lock:
                    audio_playing = True
                
                try:
                    with TimingContext("audio_generation_total"):
                        # Generate audio
                        with TimingContext("gtts_generation"):
                            audio_file = os.path.join(AUDIO_DIR, f"caption_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp3")
                            tts = gTTS(text=caption, lang='en')
                            tts.save(audio_file)
                        
                        # Play audio
                        with TimingContext("audio_playback"):
                            playsound(audio_file)
                        
                        # Cleanup
                        with TimingContext("audio_cleanup"):
                            os.remove(audio_file)
                            
                finally:
                    with audio_control_lock:
                        audio_playing = False
                        
            time.sleep(0.1)
            
        except Exception as e:
            print(f"Error in audio thread: {e}")
            with audio_control_lock:
                audio_playing = False
                
    print("Audio thread exiting...")

def setup_logging_and_output(output_dir):
    """Setup logging directory"""
    with TimingContext("setup_logging"):
        os.makedirs(output_dir, exist_ok=True)
        return output_dir

def initialize_realsense_camera():
    """Initialize RealSense camera with timing"""
    with TimingContext("realsense_init"):
        pipeline = rs.pipeline()
        config = rs.config()
        
        # Enable streams
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 60)
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 60)
        
        # Start pipeline
        profile = pipeline.start(config)
        
        # Get camera intrinsics
        color_stream = profile.get_stream(rs.stream.color)
        intrinsics = color_stream.as_video_stream_profile().get_intrinsics()
        
        # Create align object
        align = rs.align(rs.stream.color)
        
        print(f"Camera initialized: {intrinsics.width}x{intrinsics.height}")
        print(f"Focal length: fx={intrinsics.fx:.2f}, fy={intrinsics.fy:.2f}")
        
        return pipeline, align, intrinsics

def prepare_realsense_frames(frames, align):
    """Prepare RealSense frames with timing"""
    with TimingContext("prepare_frames"):
        aligned_frames = align.process(frames)
        
        color_frame = aligned_frames.get_color_frame()
        depth_frame = aligned_frames.get_depth_frame()
        
        if not color_frame or not depth_frame:
            return None, None, None
        
        # Convert to numpy arrays
        color_image = np.asanyarray(color_frame.get_data())
        depth_image = np.asanyarray(depth_frame.get_data())
        
        # Apply colormap to depth
        depth_colormap = cv2.applyColorMap(
            cv2.convertScaleAbs(depth_image, alpha=0.03), 
            cv2.COLORMAP_JET
        )
        
        return color_image, depth_colormap, depth_frame

def process_detections_realsense(results, color_image, depth_frame, annotated_rgb, annotated_depth, camera_intrinsics):
    """Process YOLO detections with timing"""
    with TimingContext("process_detections", metadata={'num_results': len(results)}):
        labels = []
        
        for r in results:
            boxes = r.boxes
            
            for box in boxes:
                # Get bounding box coordinates
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                
                # Get class and confidence
                cls = int(box.cls[0])
                cls_name = r.names[cls]
                confidence = float(box.conf[0]) * 100
                
                # Get depth
                with TimingContext("depth_measurement"):
                    depth_value = depth_frame.get_distance(cx, cy)
                
                # Calculate FOV
                with TimingContext("fov_calculation"):
                    fx = camera_intrinsics.fx
                    dx = cx - camera_intrinsics.ppx
                    theta_x = math.degrees(math.atan2(dx, fx))
                
                # Create label
                if depth_value == 0:
                    label = f"Class: {cls_name} | Confidence: {confidence:.2f}% | Depth: Invalid | Horizontal FOV: {theta_x:.2f}deg"
                else: 
                    label = f"Class: {cls_name} | Confidence: {confidence:.2f}% | Depth: {depth_value:.3f}m | Horizontal FOV: {theta_x:.2f}deg"
                
                labels.append(label)
                
                # Draw on both images
                with TimingContext("draw_annotations"):
                    for img in [annotated_rgb, annotated_depth]:
                        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
        return labels if labels else ["No objects detected!"]

def save_frame(output_dir, combined, time_step, labels, start_time):
    """Save frame and log information with timing"""
    with TimingContext("save_frame"):
        print(f"DEBUG: PATH length = {len(PATH)}")
        print(f"DEBUG: PATH content = {PATH}")
        frame_path = os.path.join(output_dir, f"frame_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg")
        cv2.imwrite(frame_path, combined)
        print(f"Saved: {frame_path}")
        srt_file(in_filename, "NA", labels, start_time)
        return labels

def add_captions(frame, text):
    """Add captions to frame"""
    if not text:
        return frame
        
    with TimingContext("add_captions"):
        font = cv2.FONT_HERSHEY_SIMPLEX
        org = (50, 50)
        font_scale = 0.7
        font_color = (0, 255, 0)
        thickness = 2
        line_type = cv2.LINE_AA

        (text_width, text_height) = cv2.getTextSize(text, font, font_scale, thickness)
        y_pos = frame.shape[0] - 10

        cv2.rectangle(frame, (0, y_pos - text_height - 10), (frame.shape[1], y_pos + 10), (0, 0, 0), -1)
        cv2.putText(frame, text, (10, y_pos - 5), font, font_scale, font_color, thickness)

    return frame

def run_with_depth(pipeline, align, model, output_dir, camera_intrinsics, writer):
    """Main processing loop with timing"""
    global current_caption, caption_lock
    frame_num = 0
    time_step = 0
    
    try:
        while True:
            loop_start = time.perf_counter()
            
            # Get frames
            with TimingContext("wait_for_frames"):
                frames = pipeline.wait_for_frames()
            
            # Prepare frames
            color_image, depth_colored, depth_frame = prepare_realsense_frames(frames, align)
            if color_image is None or depth_colored is None:
                continue
            
            # Run object detection
            with TimingContext("yolo_inference"):
                results = model(source=color_image, conf=0.69)
            
            annotated_rgb = color_image.copy()
            annotated_depth = depth_colored.copy()
            
            start_time = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
            
            # Process detections
            labels = process_detections_realsense(results, color_image, depth_frame, 
                                                annotated_rgb, annotated_depth, 
                                                camera_intrinsics)
            
            # Get captions
            with TimingContext("caption_lock_access"):
                with caption_lock:
                    caption_text = current_caption
                    print(f"Current Caption Text: {caption_text}")
            
            # Get expected dimensions from writer
            expected_width = 640 * 2
            expected_height = 480

            # Display combined image
            with TimingContext("frame_composition"):
                combined = np.hstack((annotated_rgb, annotated_depth))
                if combined.shape[1] != expected_width or combined.shape[0] != expected_height:
                    combined = cv2.resize(combined, (expected_width, expected_height))

                if combined.shape[2] == 3:
                    frame_to_write = cv2.cvtColor(combined, cv2.COLOR_RGB2BGR)
                else:
                    frame_to_write = combined
                    
                frame_to_write = add_captions(frame_to_write, caption_text)
            
            # Display and write
            with TimingContext("frame_display"):
                cv2.imshow("RealSense Camera (Color + Depth)", frame_to_write)
            
            with TimingContext("video_write"):
                writer.write(frame_to_write)

            # Handle frame saving and logging
            frame_num += 1
            if frame_num % 3 == 0:
                try:
                    with TimingContext("frame_queue_put"):
                        FRAME_Q.put({
                            'timestamp': start_time, 
                            'labels': labels, 
                            'combined': combined.copy()
                        }, block=False)
                except:
                    print("Frame queue is full, skipping frame.")
            
            # Calculate total loop time
            loop_duration = (time.perf_counter() - loop_start) * 1000
            print(f"🔄 Total loop: {loop_duration:.2f}ms")
            
            if chr(cv2.waitKey(1) & 0xFF) == 'q':
                break
                
    finally:
        pipeline.stop()
        writer.release()
        cv2.destroyAllWindows()

def main():
    global processing_active, camera_active
    
    print("=" * 60)
    print("ROBOT GUIDE DOG - TIMING ANALYSIS MODE")
    print("=" * 60)
    
    for directory in [BASE_OUTPUT_DIR, OUT_DIR, IN_DIR, AUDIO_DIR, VIDEO_DIR, SCREENSHOT_DIR, PICKLE_DIR]:
        os.makedirs(directory, exist_ok=True)

    vid_output_dir = setup_logging_and_output(SCREENSHOT_DIR)
    
    # Clean Folder
    for filename in os.listdir(vid_output_dir):
        if filename.endswith(".jpg"):
            os.remove(os.path.join(vid_output_dir, filename))

    print("Loading YOLO model...")
    with TimingContext("yolo_model_load"):
        model = YOLO('yolov8n.pt')

    print("Starting frame processing thread...")
    process_thread = threading.Thread(target=frame_thread_worker, daemon=True)
    process_thread.start()
    
    print("Starting audio thread worker...")
    audio_thread = threading.Thread(target=audio_thread_worker, daemon=True)
    audio_thread.start()

    # Initialize RealSense
    print("Initializing RealSense camera...")
    pipeline, align, intrinsics = initialize_realsense_camera()
    
    # Initialize video writer
    print("Initializing video writer...")
    writer, video_filename = video_init_realsense()
    
    try:
        print("Starting processing with RealSense...")
        print("Press 'q' to quit and save timing data")
        run_with_depth(pipeline, align, model, vid_output_dir, intrinsics, writer)

    except Exception as e:
        print(f"Error during processing: {e}")
    finally:
        # Cleanup
        print("\nShutting down...")
        processing_active = False
        camera_active = False
        process_thread.join(timeout=3.0)
        audio_thread.join(timeout=3.0)

        if process_thread.is_alive():
            print("Warning: Process thread did not terminate cleanly.")
        
        # Save timing data
        print("\nSaving timing data...")
        save_timing_data()
        
        print("\n" + "=" * 60)
        print(f"Total timing measurements collected: {len(TIMING_DATA)}")
        print("=" * 60)

if __name__ == "__main__":
    main()
