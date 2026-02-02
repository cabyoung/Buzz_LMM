#: realcam.py
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
#from moviepy import VideoFileClip, AudioFileClip, CompositeAudioClip, concatenate_audioclips
# from tf_transformations import euler_from_quaternion
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
# Needed: Overlaying Audio over video:full recording from beginning of thread and then overlaying audio on the video
# Timing/Intervals of Descriptions

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
    # PATH = check_path()
    if not PATH:
        PATH = ["No path data available."]
    with open(filename, 'a', encoding='utf-8') as f:
        f.write(f"Timestamp: {start}\n")
        f.write(f"Caption: {current_caption}\n")
        f.write(f"Object detection log: {log}\n")
        #f.write(f"path: {PATH}\n")
        f.write("\n")
        API_BUFF.append(f"Timestamp: {start} | Caption: {current_caption} | Object detection log: {log}") 
                        #path: {PATH}")

def srt_out(filename, current_caption="NA", log=""):
    global caption_history
    with open(filename, 'a', encoding='utf-8') as f:
        f.write(f"Object detection log: {log}\n")
        f.write(f"Caption: {current_caption}\n")
        f.write(f"Claude API BUFF: {API_BUFF}\n")
        f.write(f"Caption History: {caption_history}\n")
        f.write("\n")

def video_init_realsense():
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
"""
def claude(label):
    prompt = "You are given a number of seconds worth of data from three consecutive timestamps. Using this data, tell the human why you are moving a certain way and what you are avoiding. Also, mention any objects in the frames and their distance from the human. Let the person know if the given path must change due to obstacles." \
        " Keep the response concise and relevant to the human's navigation."
    system_prompt = "Responses need to be 8 words maximum and written in the point of view of a robot guide dog. The robot guide dog is helping a human to navigate around obstacles in the space. " \
    "Make sure your response helps the human understand why you are moving a certain way. Do not give specific information about the location of the objects. For example, if the depth data is 0.5m, do not say 'the object is 0.5m away'. Instead, say something like 'the object is close to you'. " \
    "Do not mind small changes in the FOV or depth data. Pay attention to and comment on larger changes in data among frame data given to you"
    # consecutive timestamp and a path giving future trajectory data for the robot.
    try:
        # First test network connectivity
        print("Testing network connectivity to Claude API...")
        import socket
        import requests

        # Test DNS resolution
        try:
            socket.gethostbyname('api.anthropic.com')
            print("✓ DNS resolution successful")
        except socket.gaierror as e:
            print(f"✗ DNS resolution failed: {e}")
            return "Network error: Cannot resolve API endpoint"

        # Test API endpoint reachability
        try:
            test_response = requests.get('https://api.anthropic.com', timeout=10)
            print("✓ API endpoint is reachable")
        except requests.exceptions.RequestException as e:
            print(f"API endpoint unreachable: {e}")
            return "Network error: Cannot reach Claude API"

        # Now try the actual API call
        print("Making Claude API request...")
        response = CLIENT.messages.create(
            #model="claude-sonnet-4-5-20250929",
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
        error_msg = f"Network connection error: {e}"
        logging.error(error_msg)
        print(f"✗ {error_msg}")
        return "Connection error: Cannot reach Claude service"

    except anthropic.APIStatusError as e:
        error_msg = f"API error (status {e.status_code}): {e}"
        logging.error(error_msg)
        print(f"✗ {error_msg}")
        if e.status_code == 401:
            return "API key error: Invalid authentication"
        elif e.status_code == 429:
            return "API limit exceeded: Too many requests"
        else:
            return f"API error: {e.status_code}"

    except anthropic.AuthenticationError as e:
        error_msg = f"Authentication error: {e}"
        logging.error(error_msg)
        print(f"✗ {error_msg}")
        return "Authentication failed: Check API key"

    except Exception as e:
        error_msg = f"Unexpected error: {e}"
        logging.error(error_msg)
        print(f"✗ {error_msg}")
        return "Service temporarily unavailable"

"""    

def claude(label):
    global latest_caption, caption_history
    prompt = "Navigation update (8 words max)"
    system_prompt = f"You are a robot guide dog. You are given 3 seconds of data from three consecutive timestamps." \
    f"FOV is from the robot's perspective which has a total horizontal FOV = 87 degrees." \
    f"Positive angle in FOV = object is to the right of center, negative angle in FOV = object is to the left of center." \
    # f"FOV is from your perspective:more negative angle due to object = you are moving to the right and more positive angle due to object = you are moving to the left." \
    f"Using this data, and the captions from the consecutive previous seconds for context to tell the human about your movement: {caption_history}"
    #"due to objects and the path." 
    try:
        response = "" 
        with CLIENT.messages.stream(
            model="claude-sonnet-4-5-20250929",
            max_tokens=50,
            system=system_prompt,
            messages=[{
                "role": "user",
                "content": label + "\n" + prompt
            }]
        ) as stream:
            for text in stream.text_stream:
                response += text

        return response.strip()
    
    except anthropic.APIConnectionError as e:
        logging.error(f"Connection error: {e}")
        return "Navigation active"
    except anthropic.APIStatusError as e:
        logging.error(f"API error: {e}")
        return "Processing..."
    except anthropic.APIError as e:  # Add this - catches other Anthropic errors
        logging.error(f"Anthropic API error: {e}")
        return "Active"

def sound_check(response, filename="sound_check.mp3"):
    language = 'en'
    try:
        tts = gTTS(text=response, lang=language, slow=False)
        # Save the audio file and add to array to combine later
        audio_path = os.path.join(AUDIO_DIR, filename)
        VOICE.append(audio_path)
        tts.save(audio_path)
        playsound(audio_path)
        print("Audio played successfully.")
    except Exception as e:
        print(f"Error playing sound: {e}")
        print("You might need to install an audio player compatible with 'playsound' or check file path.")

def sound_check_thread(label, thread_id):
    global current_caption
    global caption_lock
    print(f"[{thread_id}] Starting sound generation and playback for: {label[:50]}...")
    try:
        claude_response = claude(label)
        print(f"[{thread_id}] Claude Response: {claude_response}")
        with caption_lock:
            current_caption = claude_response  # Update the global caption variable
        # Use a unique filename for each sound instance to prevent conflicts
        audio_filename = f"sound_{thread_id}.mp3"
        sound_check(claude_response, filename=audio_filename)
    except Exception as e:
        logging.error(f"[{thread_id}] An error occurred in sound thread: {e}")
    print(f"[{thread_id}] Sound thread finished.")

def sound_worker_thread():
    global sound_thread_active
    global current_caption
    global caption_lock
    print("Sound worker thread started.")
    while sound_thread_active:
        # Wait for a signal that there's a new sound request
        sound_thread_event.wait(timeout=1.0) # Wait up to 1 second

        # Clear the event so it can be set again for the next request
        if sound_thread_event.is_set():
            sound_thread_event.clear()

        # Process all requests currently in the queue
        while sound_request_queue:
            label_text = sound_request_queue.popleft() # Get the oldest request
            thread_id = time.time() # Use timestamp for unique ID

            print(f"[Sound Worker] Processing: {label_text[:50]}...")
            try:
                claude_response = claude(label_text)
                print(f"[Sound Worker] Claude Response: {claude_response}")
                with caption_lock:
                    current_caption = claude_response
                audio_filename = f"sound_{int(thread_id)}.mp3" # Unique filename
                sound_check(claude_response, filename=audio_filename)
            except Exception as e:
                logging.error(f"[Sound Worker] An error occurred: {e}")
            print(f"[Sound Worker] Finished processing request.")

    print("Sound worker thread stopped.")

def audio_thread_worker():
    global latest_caption, audio_playing, audio_control_lock, processing_active, current_caption, caption_lock
    print("Audio thread worker started.")
    while processing_active:
        try:
            with audio_control_lock:
                if latest_caption is not None and not audio_playing:
                    caption_to_play = latest_caption
                    latest_caption = None  # Clear after taking it
                    audio_playing = True
                else:
                    caption_to_play = None
            if caption_to_play:
                with caption_lock:
                    current_caption = caption_to_play  # Update the global caption variable
                print(f"Playing audio for caption: {caption_to_play}")
                sound_check(caption_to_play, f"sound_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp3")
                caption_history.clear()
                caption_history.append(caption_to_play)
                print("Audio playback finished.")
                with audio_control_lock:
                    audio_playing = False
            else:
                time.sleep(0.1)  # Sleep briefly to avoid busy waiting
        except Exception as e:
            print(f"Error in audio thread: {e}")
            with audio_control_lock:
                audio_playing = False
    print("Audio thread worker stopped.")

def frame_thread_worker():
    global processing_active, current_caption, caption_lock
    global latest_caption, audio_control_lock, PATH
    time_step = 0
    print("Frame thread worker started.")
    while processing_active:
        try:
            # Simulate frame processing
            frame_data = FRAME_Q.get(timeout=1.0)  # Wait for a frame
            print(f"[Frame Worker] Got frame from queue")
            save_frame(SCREENSHOT_DIR, frame_data['combined'], time_step, frame_data['labels'], frame_data['timestamp'])
            print(f"[Frame Worker] API_BUFF length: {len(API_BUFF)}")
            if len(API_BUFF) == 3:
                batch_data = "\n".join(API_BUFF)
                PATH = check_path()  # Update PATH before sending to Claude
                # Get caption from Claude
                # Add time
                caption = claude(batch_data)
                # Store caption
                caption_history.append(caption)
                #with caption_lock:
                #    current_caption = caption
                #    print(f"Current Caption: {current_caption}")
                # May need Change
                srt_out(out_filename, caption, API_BUFF.copy())
                    # srt_file(out_filename, caption_text, label, start_time, end_time)
                with audio_control_lock:
                    latest_caption = caption  # Update latest caption for audio thread
                    print(f"New caption ready: {caption}")
                    if audio_playing:
                        print("Audio is currently playing, new caption will wait.")
                #audio_thread = threading.Thread(target=lambda: sound_check(caption, f"sound_{time.time()}.mp3"), daemon=True)
                #audio_thread.start()

                API_BUFF.clear()
                time_step += 1

        except Empty:
            continue  # Timeout waiting for frame, loop again

        except Exception as e:
            if type(e).__nameq__ == 'Empty':
                import traceback
                print(f"\n{'='*60}")
                print(f"ERROR in frame_processing_thread:")
                print(f"Exception type: {type(e).__name__}")
                print(f"Exception message: {str(e)}")
                print(f"Full traceback:")
                traceback.print_exc()
                print(f"{'='*60}\n")
                continue
    print("Frame thread worker stopped.")

def setup_logging_and_output(output_dir):
    """Initialize logging and create output directory"""
    os.makedirs(SCREENSHOT_DIR, exist_ok=True)
    logging.basicConfig(
        filename= os.path.join(BASE_OUTPUT_DIR, "test.txt"), 
        filemode="w", 
        level=logging.INFO, 
        format="%(asctime)s - %(message)s", 
        datefmt="%H:%M:%S"
    )
    return SCREENSHOT_DIR

def initialize_realsense_camera():
    """Initialize and configure Intel RealSense D435"""
    print("Setting up Intel RealSense D435...")
    
    # Configure depth and color streams
    pipeline = rs.pipeline()
    config = rs.config()
    
    # Enable streams
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    
    # Start streaming
    profile = pipeline.start(config)
    
    # Get camera intrinsics
    color_profile = profile.get_stream(rs.stream.color)
    intrinsics = color_profile.as_video_stream_profile().get_intrinsics()
    camera_intrinsics = (intrinsics.fx, intrinsics.fy, intrinsics.ppx, intrinsics.ppy)
    
    # Align depth to color
    align_to = rs.stream.color
    align = rs.align(align_to)
    
    return pipeline, align, camera_intrinsics

def prepare_realsense_frames(frames, align):
    """Process RealSense frames to get aligned color and depth images"""
    # Align the depth frame to color frame
    aligned_frames = align.process(frames)
    
    # Get aligned frames
    depth_frame = aligned_frames.get_depth_frame()
    color_frame = aligned_frames.get_color_frame()
    
    if not depth_frame or not color_frame:
        return None, None, None
    
    # Convert images to numpy arrays
    color_image = np.asanyarray(color_frame.get_data())
    depth_image = np.asanyarray(depth_frame.get_data())
    
    # Process depth for visualization
    depth_colored = cv2.applyColorMap(
        cv2.convertScaleAbs(depth_image, alpha=0.03), 
        cv2.COLORMAP_JET
    )
    
    return color_image, depth_colored, depth_frame

def process_detections_realsense(results, color_image, depth_frame, annotated_rgb, annotated_depth, camera_intrinsics):
    """Process YOLO detections with RealSense depth data"""
    fx, fy, px, py = camera_intrinsics
    labels = []
    
    for result in results:
        if len(result.boxes) > 0:
            for box_idx in range(len(result.boxes)):
                x1, y1, x2, y2 = map(int, result.boxes.xyxy[box_idx].cpu().numpy())
                
                # Get detection information
                cls_id = int(result.boxes.cls[box_idx]) #detection.boxes
                cls_name = result.names[cls_id] #results[0].names
                confidence = result.boxes.conf[box_idx] * 100
                center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
                
                # Get depth value (convert from mm to meters)
                depth_value = depth_frame.get_distance(center_x, center_y) if depth_frame else 0
                
                # Angles
                theta_x = math.degrees(math.atan2((center_x - px), fx))
                theta_y = math.degrees(math.atan2((center_y - py), fy))
                
                # Create label
                if depth_value == 0 or math.isinf(depth_value):
                    label = f"Class: {cls_name} | Confidence: {confidence:.2f}% | Depth: Invalid | Horizontal FOV: {theta_x:.2f}deg"
                else: 
                    label = f"Class: {cls_name} | Confidence: {confidence:.2f}% | Depth: {depth_value:.3f}m | Horizontal FOV: {theta_x:.2f}deg"
                
                labels.append(label)
                
                # Draw on both images
                for img in [annotated_rgb, annotated_depth]:
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    return labels if labels else ["No objects detected!"]

def save_frame(output_dir, combined, time_step, labels, start_time):
    """Save frame and log information"""
    print(f"DEBUG: PATH length = {len(PATH)}")
    print(f"DEBUG: PATH content = {PATH}")
    frame_path = os.path.join(output_dir, f"frame_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg")
    cv2.imwrite(frame_path, combined) # Save Frame
    print(f"Saved: {frame_path}")
    srt_file(in_filename, "NA", labels, start_time)
    """
    try:
        for label in labels:
            srt_file(in_filename, "NA", label, start_time, end_time)
    except Exception as e:
        print(f"Error saving SRT file: {e}")
    """
    return labels

def add_captions(frame, text):
    if not text:
        return frame
    # Add Captions
    print(f"Adding caption: {text}")
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
    global current_caption, caption_lock
    frame_num = 0
    time_step = 0
    # check_time = 3.0
    # last_time = time.time()
    
    try:
        while True:
            frames = pipeline.wait_for_frames()
            # Prepare frames
            color_image, depth_colored, depth_frame = prepare_realsense_frames(frames, align)
            if color_image is None or depth_colored is None:
                continue
            
            # Run object detection
            results = model(source=color_image, conf=0.69)
            annotated_rgb = color_image.copy()
            annotated_depth = depth_colored.copy()
            
            start_time = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
            
            # Process detections
            labels = process_detections_realsense(results, color_image, depth_frame, 
                                                annotated_rgb, annotated_depth, 
                                                camera_intrinsics)
            
            # Captions
            with caption_lock:
                caption_text = current_caption  # Use the global caption variable
                print(f"Current Caption Text: {caption_text}")
            
            #end_time = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            # Get expected dimensions from writer
            expected_width = 640 * 2  # Double width for side-by-side
            expected_height = 480

            # Display combined image
            combined = np.hstack((annotated_rgb, annotated_depth))
            if combined.shape[1] != expected_width or combined.shape[0] != expected_height:
                combined = cv2.resize(combined, (expected_width,expected_height))

            if combined.shape[2] == 3:
                frame_to_write = cv2.cvtColor(combined, cv2.COLOR_RGB2BGR)
            else:
                frame_to_write = combined  
            frame_to_write = add_captions(frame_to_write, caption_text)  # Add captions to frame
            cv2.imshow("RealSense Camera (Color + Depth)", frame_to_write)
            writer.write(frame_to_write)  # Save to video file

            # Handle frame saving and logging
            frame_num += 1
            # Queue every 3rd frame for processing
            if frame_num % 3 == 0:
                try: 
                    FRAME_Q.put({'timestamp': start_time, 'labels': labels, 'combined': combined.copy()}, block=False)
                    #label_for_sound = save_frame(SCREENSHOT_DIR, combined, time_step, labels, start_time)
                    # SECONDS OF FRAMES
                except:
                    print("Frame queue is full, skipping frame.")
            
            if chr(cv2.waitKey(1) & 0xFF) == 'q':
                break
    finally:
        pipeline.stop()
        writer.release()
        cv2.destroyAllWindows()
        """
                    if len(API_BUFF) == 5:
                        for log in HIST_BUFF:
                            srt_out(out_filename, caption_text, log)
                            # srt_file(out_filename, caption_text, label, start_time, end_time)
                            HIST_BUFF.pop()
                        
                        # Add the request to the queue and signal the sound worker
                        sound_request_queue.append("\n".join(API_BUFF))
                        sound_thread_event.set() # Signal the sound thread
                        # add the 5 to history
                        HIST_BUFF.append(API_BUFF.copy())
                        API_BUFF.clear()
                        frame_num = 0
                        time_step += 1
                except Exception as e:
                    print(f"Error during frame save/log: {e}")
                    continue
            # Check for exit key
            if chr(cv2.waitKey(1) & 0xFF) == 'q':
                break
        
    finally:
        pipeline.stop()
        writer.release()
        cv2.destroyAllWindows()
        """

def main():
    global processing_active, camera_active
    for directory in [BASE_OUTPUT_DIR, OUT_DIR, IN_DIR, AUDIO_DIR, VIDEO_DIR, SCREENSHOT_DIR]:
        os.makedirs(directory, exist_ok=True)

    # global sound_thread_active, camera_active
    vid_output_dir = setup_logging_and_output(SCREENSHOT_DIR)
    # Clean Folder
    for filename in os.listdir(vid_output_dir):
        if filename.endswith(".jpg"):
            os.remove(os.path.join(vid_output_dir, filename))

    print("Loading YOLO model...")
    model = YOLO('yolov8n.pt')
    
    # Start sound worker thread
    # print("Starting sound worker thread...")
    # sound_thread = threading.Thread(target=sound_worker_thread)
    # sound_thread.daemon = True
    # sound_thread.start()

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
        run_with_depth(pipeline, align, model, vid_output_dir, intrinsics, writer)

    except Exception as e:
        print(f"Error during processing: {e}")
    finally:
        # Cleanup
        processing_active = False
        # sound_thread_event.set()
        camera_active = False
        # sound_thread.join(timeout=5.0)
        process_thread.join(timeout=3.0)
        audio_thread.join(timeout=3.0)

        if process_thread.is_alive():
            print("Warning: Process thread did not terminate cleanly.")

if __name__ == "__main__":
    main()