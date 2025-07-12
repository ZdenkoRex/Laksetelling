import os
import cv2
import time
import subprocess
import numpy as np
from datetime import datetime
from inference import InferencePipeline

# === 🔧 KONFIGURASJON ===
API_KEY = "BbEnnxyJOjvhcNRsZlSi"
WORKSPACE = "laksetelling"
WORKFLOW_ID = "detect-count-and-visualize-2"
VIDEO_REFERENCE = "Testvideo.mov"  # Kun for init – brukes ikke
STREAM_URL = "https://denali.hi.no/webkamera/smil:etne2023.smil/playlist.m3u8"
OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 🎯 Parametre for sporing
IOU_THRESHOLD = 0.3
MIN_SECONDS_BETWEEN_DETECTIONS = 2
frame_width, frame_height = 1280, 720

fish_id_counter = 0
tracked_fish = []

# === 📦 Funksjoner ===
def calculate_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    if interArea == 0:
        return 0.0
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    return interArea / float(boxAArea + boxBArea - interArea)

def save_snapshot(frame, label, x1, y1, x2, y2):
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = os.path.join(OUTPUT_DIR, f"{label}_{timestamp}.jpg")
    cropped = frame[y1:y2, x1:x2]
    cv2.imwrite(filename, cropped)
    print(f"📸 Lagret bilde: {filename}")

def is_new_fish(box, timestamp):
    for fish in tracked_fish:
        if calculate_iou(box, fish["box"]) > IOU_THRESHOLD:
            if timestamp - fish["last_seen"] < MIN_SECONDS_BETWEEN_DETECTIONS:
                return False
    return True

def update_tracking(box, fish_id, timestamp):
    for fish in tracked_fish:
        if fish["id"] == fish_id:
            fish["box"] = box
            fish["last_seen"] = timestamp
            return
    tracked_fish.append({"id": fish_id, "box": box, "last_seen": timestamp})

def my_sink(result, video_frame):
    global fish_id_counter
    timestamp = time.time()

    try:
        frame = video_frame.image
        detections = result.get("predictions", None)
        if detections is None or len(detections.xyxy) == 0:
            return

        for i, bbox in enumerate(detections.xyxy):
            x1, y1, x2, y2 = map(int, bbox)
            class_name = (
                detections.data["class_name"][i]
                if "class_name" in detections.data and len(detections.data["class_name"]) > i
                else "salmon"
            )

            box = (x1, y1, x2, y2)

            if is_new_fish(box, timestamp):
                fish_id_counter += 1
                label = f"{class_name} #{fish_id_counter}"
                save_snapshot(frame, label, x1, y1, x2, y2)
                update_tracking(box, fish_id_counter, timestamp)
            else:
                # Finn korrekt ID basert på overlapping
                for fish in tracked_fish:
                    if calculate_iou(box, fish["box"]) > IOU_THRESHOLD:
                        label = f"{class_name} #{fish['id']}"
                        update_tracking(box, fish["id"], timestamp)
                        break

            # Tegn
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.imshow("🐟 Laksetelling", frame)
        cv2.waitKey(1)

    except Exception as e:
        print(f"❌ Feil i sink: {e}")

# === 🚀 Start pipeline ===
print("🎯 Initialiserer Roboflow pipeline...")
pipeline = InferencePipeline.init_with_workflow(
    api_key=API_KEY,
    workspace_name=WORKSPACE,
    workflow_id=WORKFLOW_ID,
    video_reference=STREAM_URL,
    max_fps=30,
    on_prediction=my_sink
)
pipeline.start()

# === 📺 FFmpeg livestream ===
print("🔗 Åpner livestream...")
ffmpeg_cmd = [
    "ffmpeg",
    "-i", STREAM_URL,
    "-loglevel", "quiet",
    "-an",
    "-f", "image2pipe",
    "-pix_fmt", "bgr24",
    "-vcodec", "rawvideo",
    "-"
]

ffmpeg_process = subprocess.Popen(
    ffmpeg_cmd,
    stdout=subprocess.PIPE,
    bufsize=10**8
)

# === 🔄 Prosesser hver ramme ===
try:
    while True:
        raw_frame = ffmpeg_process.stdout.read(frame_width * frame_height * 3)
        if not raw_frame:
            print("❌ Mistet forbindelse til stream.")
            break

        frame = np.frombuffer(raw_frame, dtype=np.uint8).reshape((frame_height, frame_width, 3))
        pipeline.enqueue_frame(frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            print("🛑 Avslutter...")
            break

except KeyboardInterrupt:
    print("⛔️ Stoppet med Ctrl+C.")
finally:
    ffmpeg_process.terminate()
    pipeline.join()
    cv2.destroyAllWindows()
