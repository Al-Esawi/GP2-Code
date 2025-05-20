import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
import cv2
import numpy as np
import tensorflow as tf
from collections import deque, Counter
import threading
import os
import sys
import random
import time
import pygame
import mediapipe as mp
import traceback

# ------------------ Exception Hook ------------------
def except_hook(exctype, value, tb):
    print("[EXCEPTION]", value)
    traceback.print_exception(exctype, value, tb)
    input("Press Enter to exit...")

sys.excepthook = except_hook

# ------------------ MediaPipe Setup ------------------
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, min_detection_confidence=0.5)

# ------------------ Face Cropping ------------------
def crop_face(image):
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_image)

    if results.multi_face_landmarks:
        landmarks = results.multi_face_landmarks[0]
        h, w, _ = image.shape
        min_x = min([landmark.x * w for landmark in landmarks.landmark])
        min_y = min([landmark.y * h for landmark in landmarks.landmark])
        max_x = max([landmark.x * w for landmark in landmarks.landmark])
        max_y = max([landmark.y * h for landmark in landmarks.landmark])

        margin = 0.15
        x_exp = int(max(min_x - margin * w, 0))
        y_exp = int(max(min_y - margin * h, 0))
        x2_exp = int(min(max_x + margin * w, w))
        y2_exp = int(min(max_y + margin * h, h))

        cropped_face = image[y_exp:y2_exp, x_exp:x2_exp]
        if cropped_face.size > 0:
            return cv2.resize(cropped_face, (224, 224), interpolation=cv2.INTER_AREA)
    return None

# ------------------ Preprocessing ------------------
def preprocess(frame):
    if frame is None or frame.size == 0:
        return None
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = cv2.resize(frame, (224, 224), interpolation=cv2.INTER_AREA)
    frame = frame.astype(np.float32) / 255.0
    return np.expand_dims(frame, axis=0)

# ------------------ Consent Form ------------------
def show_consent_form():
    consent_result = {"agreed": False}
    def agree():
        consent_result["agreed"] = True
        consent_window.destroy()
    def decline():
        messagebox.showerror("Consent Required", "You must agree to proceed.")
        consent_window.destroy()
    consent_window = tk.Tk()
    consent_window.title("Consent Form")
    consent_window.geometry("400x300")
    consent_window.resizable(False, False)
    tk.Label(consent_window, text="Consent Form", font=("Arial", 16, "bold")).pack(pady=10)
    consent_text = (
        "By using this software, you consent to the collection of emotional data for educational purposes. "
        "This data is used solely for improving user engagement with the Virtual Study Companion system."
    )
    tk.Label(consent_window, text=consent_text, font=("Arial", 11), wraplength=350, justify="left").pack(padx=20, pady=20)
    button_frame = tk.Frame(consent_window)
    button_frame.pack(pady=10)
    ttk.Button(button_frame, text="I Agree", command=agree).pack(side="left", padx=10)
    ttk.Button(button_frame, text="Cancel", command=decline).pack(side="left", padx=10)
    consent_window.mainloop()
    return consent_result["agreed"]

# ------------------ Resource Path ------------------
def resource_path(relative_path):
    try:
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)

# ------------------ AudioPlayer Class ------------------
class AudioPlayer:
    def __init__(self):
        pygame.mixer.init(frequency=22050, size=-16, channels=2)
        self.stop_event = threading.Event()
        self.thread = None
        self.loop_count = 0
        self.max_loops = 1

    def play_loop(self, audio_path):
        self.stop()
        self.stop_event.clear()
        self.max_loops = 3 if "looking_away" in audio_path.lower() else float('inf')
        self.thread = threading.Thread(target=self._play_loop_thread, args=(audio_path,), daemon=True)
        self.thread.start()

    def _play_loop_thread(self, audio_path):
        try:
            sound = pygame.mixer.Sound(audio_path)
            self.loop_count = 0
            while not self.stop_event.is_set() and self.loop_count < self.max_loops:
                sound.play()
                self.loop_count += 1
                while pygame.mixer.get_busy() and not self.stop_event.is_set():
                    time.sleep(0.1)
        except Exception as e:
            print("[DEBUG] Audio error:", e)

    def stop(self):
        self.stop_event.set()
        if self.thread and self.thread.is_alive():
            self.thread.join()
        pygame.mixer.stop()

    def pause(self):
        pygame.mixer.pause()

    def resume(self):
        pygame.mixer.unpause()

# ------------------ EmotionApp Class ------------------
class EmotionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Virtual Study Companion")
        self.root.geometry("700x500")
        self.root.configure(bg="#f2f2f2")
        self.root.resizable(False, False)

        model_path = resource_path(r"C:\Users\user\Desktop\project\updated_mesh_model")
        try:
            self.model = tf.keras.models.load_model(model_path)
        except Exception as e:
            messagebox.showerror("Model Error", f"Could not load the model: {e}")
            sys.exit()

        self.labels = ['Engaged', 'Looking Away', 'bored', 'confused', 'drowsy', 'frustrated']
        audio_dir = r"C:\Users\user\Desktop\project\audio2"
        self.response_audio = {label: os.path.join(audio_dir, f"{label}.mp3") for label in self.labels}

        self.text_responses = {
            'confused': ["It looks like you're a bit confused.", "Feeling confused? Try reviewing the topic."],
            'Engaged': ["Great job staying focused!", "You're fully engaged!"],
            'frustrated': ["You seem frustrated. Take a break!", "Try deep breaths to calm down."],
            'Looking Away': ["Try to refocus on the screen.", "You're looking away from the task."],
            'bored': ["You look bored. Time for a quick stretch?", "Switch things up to stay interested!"],
            'drowsy': ["Feeling sleepy? Take a short walk.", "Try stretching to wake up!"]
        }

        self.emotion_emoji = {
            'confused': '😕', 'Engaged': '😃', 'frustrated': '😠',
            'Looking Away': '👀', 'bored': '😒', 'drowsy': '😴'
        }

        self.cap = None
        self.monitor_thread = None
        self.stop_event = threading.Event()
        self.audio_player = AudioPlayer()
        self.emotion_window = deque(maxlen=60)

        self.current_emotion = tk.StringVar(value="None")
        self.status = tk.StringVar(value="Idle")
        self.audio_enabled = tk.BooleanVar(value=True)

        self.is_monitoring = False

        self.build_ui()
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    def build_ui(self):
        title = tk.Label(self.root, text="🎓 Virtual Study Companion", font=("Arial", 20, "bold"), bg="#f2f2f2", fg="#333")
        title.pack(pady=10)

        frame = tk.Frame(self.root, bg="#f2f2f2")
        frame.pack(pady=5)

        tk.Label(frame, text="Current Emotion:", font=("Arial", 14), bg="#f2f2f2").grid(row=0, column=0, sticky="w")
        tk.Label(frame, textvariable=self.current_emotion, font=("Arial", 14, "bold"), fg="#007acc", bg="#f2f2f2").grid(row=0, column=1, sticky="w", padx=10)

        tk.Label(frame, text="Status:", font=("Arial", 14), bg="#f2f2f2").grid(row=1, column=0, sticky="w")
        tk.Label(frame, textvariable=self.status, font=("Arial", 14), fg="#cc6600", bg="#f2f2f2").grid(row=1, column=1, sticky="w", padx=10)

        button_frame = tk.Frame(self.root, bg="#f2f2f2")
        button_frame.pack(pady=10)

        ttk.Button(button_frame, text="▶ Start Monitoring", command=self.start_monitoring).grid(row=0, column=0, padx=10)
        ttk.Button(button_frame, text="■ Stop Monitoring", command=self.stop_monitoring).grid(row=0, column=1, padx=10)
        ttk.Checkbutton(button_frame, text="Play Audio", variable=self.audio_enabled).grid(row=0, column=2, padx=10)
        ttk.Button(button_frame, text="🔈 Pause Audio", command=self.audio_player.pause).grid(row=0, column=3, padx=10)
        ttk.Button(button_frame, text="▶️ Resume Audio", command=self.audio_player.resume).grid(row=0, column=4, padx=10)

        tk.Label(self.root, text="Emotion Log:", font=("Arial", 12, "bold"), bg="#f2f2f2").pack(anchor="w", padx=20)
        self.log_area = scrolledtext.ScrolledText(self.root, width=80, height=12, font=("Courier", 10))
        self.log_area.pack(padx=20, pady=5)

    def start_monitoring(self):
        if self.is_monitoring:
            return
        self.is_monitoring = True
        self.status.set("Monitoring...")
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)

        if not self.cap.isOpened():
            messagebox.showerror("Camera Error", "Cannot open webcam.")
            self.is_monitoring = False
            self.status.set("Idle")
            return

        self.stop_event.clear()
        self.monitor_thread = threading.Thread(target=self.monitor_loop, daemon=True)
        self.monitor_thread.start()

    def stop_monitoring(self):
        if not self.is_monitoring:
            return
        self.status.set("Stopping...")
        self.stop_event.set()
        self.audio_player.stop()
        if self.cap and self.cap.isOpened():
            self.cap.release()
        self.status.set("Stopped")
        self.is_monitoring = False

    def monitor_loop(self):
        emotion_list = []
        no_face_timer = 0

        while not self.stop_event.is_set():
            ret, frame = self.cap.read()
            if not ret:
                time.sleep(1)
                continue

            face_crop = crop_face(frame)
            if face_crop is None:
                if no_face_timer == 0:
                    self.current_emotion.set("No face detected ❌")
                    no_face_timer = time.time()
                elif time.time() - no_face_timer >= 3:
                    self.current_emotion.set("")
                time.sleep(1)
                continue
            else:
                no_face_timer = 0

            img_input = preprocess(face_crop)
            if img_input is None:
                time.sleep(1)
                continue

            prediction = self.model.predict(img_input, verbose=0)
            emotion = self.labels[np.argmax(prediction)]
            print("[DEBUG] Raw Probabilities:", prediction[0])
            print("[DEBUG] Argmax Index:", np.argmax(prediction[0]))
            print("[DEBUG] Predicted Label:", emotion)

            emotion_list.append(emotion)
            self.emotion_window.append(emotion)

            if len(emotion_list) >= 60:
                most_common_emotion = Counter(emotion_list).most_common(1)[0][0]
                self.current_emotion.set(f"{most_common_emotion} {self.emotion_emoji.get(most_common_emotion, '')}")
                self.log_area.insert(tk.END, f"{time.strftime('%H:%M:%S')} - {most_common_emotion}\n")
                self.log_area.yview_moveto(1)
                self.trigger_response(most_common_emotion)
                emotion_list.clear()

            time.sleep(2)

    def trigger_response(self, emotion):
        self.audio_player.stop()
        if self.audio_enabled.get():
            audio_path = self.response_audio.get(emotion)
            if audio_path and os.path.exists(audio_path):
                self.audio_player.play_loop(audio_path)
        responses = self.text_responses.get(emotion, [])
        if responses:
            messagebox.showinfo(f"{emotion} Detected", random.choice(responses))

    def on_close(self):
        if self.is_monitoring:
            self.stop_monitoring()
        self.root.destroy()

# ------------------ Run App ------------------
if __name__ == "__main__":
    agreed = show_consent_form()
    if agreed:
        root = tk.Tk()
        app = EmotionApp(root)
        root.mainloop()
    else:
        sys.exit()
