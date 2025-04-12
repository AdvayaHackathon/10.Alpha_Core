import cv2
import numpy as np
import time
import threading
import queue
import pyaudio
import wave
import tkinter
import tkinter as tk
from tkinter import ttk, scrolledtext
from PIL import Image, ImageTk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import os

# Create a directory for audio samples if it doesn't exist
if not os.path.exists('audio_samples'):
    os.makedirs('audio_samples')

# Mock sentiment analysis function (replace with actual sentiment analysis model if available)
def analyze_sentiment(text):
    # This is a placeholder - in a real app, you'd connect to a sentiment model
    # For demo purposes, we'll return random values
    import random
    sentiment_score = random.uniform(0, 1)
    if sentiment_score > 0.6:
        return {"label": "POSITIVE", "score": sentiment_score}
    else:
        return {"label": "NEGATIVE", "score": 1 - sentiment_score}

class AudioRecorder:
    def __init__(self, queue_size=10):
        self.audio_queue = queue.Queue(maxsize=queue_size)
        self.is_recording = False
        self.audio_thread = None
        
        # Audio parameters
        self.format = pyaudio.paInt16
        self.channels = 1
        self.rate = 44100
        self.chunk = 1024
        self.audio = pyaudio.PyAudio()
        
    def start_recording(self):
        if self.is_recording:
            return
            
        self.is_recording = True
        self.audio_thread = threading.Thread(target=self._record_audio)
        self.audio_thread.daemon = True
        self.audio_thread.start()
        
    def stop_recording(self):
        self.is_recording = False
        if self.audio_thread:
            self.audio_thread.join(timeout=1.0)
            self.audio_thread = None
            
    def _record_audio(self):
        stream = self.audio.open(
            format=self.format,
            channels=self.channels,
            rate=self.rate,
            input=True,
            frames_per_buffer=self.chunk
        )
        
        while self.is_recording:
            try:
                data = stream.read(self.chunk, exception_on_overflow=False)
                audio_data = np.frombuffer(data, dtype=np.int16)
                
                # Calculate amplitude
                amplitude = np.abs(audio_data).mean()
                
                # Put in queue if not full
                if not self.audio_queue.full():
                    self.audio_queue.put(amplitude)
            except Exception as e:
                print(f"Audio recording error: {e}")
                break
                
        stream.stop_stream()
        stream.close()
        
    def save_audio_sample(self, duration=3):
        """Records and saves a short audio sample for analysis"""
        filename = f"audio_samples/sample_{int(time.time())}.wav"
        
        stream = self.audio.open(
            format=self.format,
            channels=self.channels,
            rate=self.rate,
            input=True,
            frames_per_buffer=self.chunk
        )
        
        print(f"Recording audio sample for {duration} seconds...")
        frames = []
        for i in range(0, int(self.rate / self.chunk * duration)):
            data = stream.read(self.chunk, exception_on_overflow=False)
            frames.append(data)
            
        stream.stop_stream()
        stream.close()
        
        # Save the audio file
        wf = wave.open(filename, 'wb')
        wf.setnchannels(self.channels)
        wf.setsampwidth(self.audio.get_sample_size(self.format))
        wf.setframerate(self.rate)
        wf.writeframes(b''.join(frames))
        wf.close()
        
        print(f"Audio sample saved to {filename}")
        return filename
        
    def get_latest_amplitude(self):
        if self.audio_queue.empty():
            return 0
        
        # Get all available amplitudes and return the average
        amplitudes = []
        while not self.audio_queue.empty():
            amplitudes.append(self.audio_queue.get())
            
        if amplitudes:
            return sum(amplitudes) / len(amplitudes)
        return 0
        
    def __del__(self):
        self.stop_recording()
        self.audio.terminate()

class FaceAnalyzer:
    def __init__(self):
        # Load face detector
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        # Load eye detector
        self.eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
        
    def detect_face_and_eyes(self, frame):
        if frame is None:
            return None, None, None
            
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )
        
        if len(faces) == 0:
            return None, None, None
            
        # Use the first face found
        (x, y, w, h) = faces[0]
        face_roi = gray[y:y+h, x:x+w]
        
        # Detect eyes in the face region
        eyes = self.eye_cascade.detectMultiScale(
            face_roi,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(20, 20)
        )
        
        # Find pupil in each eye
        pupils = []
        eye_regions = []
        
        for (ex, ey, ew, eh) in eyes:
            eye_roi = face_roi[ey:ey+eh, ex:ex+ex+ew]
            eye_regions.append((ex+x, ey+y, ew, eh))
            
            # Try to find pupil using Hough Circles
            try:
                eye_roi_blur = cv2.GaussianBlur(eye_roi, (5, 5), 0)
                circles = cv2.HoughCircles(
                    eye_roi_blur,
                    cv2.HOUGH_GRADIENT,
                    dp=1,
                    minDist=20,
                    param1=50,
                    param2=30,
                    minRadius=2,
                    maxRadius=10
                )
                
                if circles is not None:
                    circles = np.uint16(np.around(circles))
                    for circle in circles[0, :]:
                        # Adjust coordinates to original frame
                        cx, cy, r = circle
                        pupils.append((ex+x+cx, ey+y+cy, r))
            except Exception as e:
                print(f"Error detecting pupils: {e}")
        
        return (x, y, w, h), eye_regions, pupils
        
    def analyze_expression(self, face_roi):
        # This is a simplistic placeholder for facial expression analysis
        # In a real app, you'd use a more advanced model
        
        if face_roi is None:
            return {"emotion": "unknown", "confidence": 0}
            
        # Simple brightness-based analysis for demonstration
        avg_brightness = np.mean(face_roi)
        
        if avg_brightness < 80:
            return {"emotion": "serious", "confidence": 0.7}
        elif avg_brightness < 120:
            return {"emotion": "neutral", "confidence": 0.6}
        else:
            return {"emotion": "bright", "confidence": 0.8}

class StressAnalyzerApp:
    def __init__(self, window, window_title):
        self.window = window
        self.window.title(window_title)
        self.window.geometry("1200x800")
        
        # Initialize components
        self.face_analyzer = FaceAnalyzer()
        self.audio_recorder = AudioRecorder()
        
        # Data storage
        self.stress_history = []
        self.sentiment_history = []
        self.audio_amplitudes = []
        self.pupil_sizes = []
        
        # Video capture
        self.cap = cv2.VideoCapture(0)
        
        # Create UI elements
        self.create_ui()
        
        # Start threads
        self.running = True
        self.audio_recorder.start_recording()
        
        # Start video processing
        self.process_video()
        
        self.window.protocol("WM_DELETE_WINDOW", self.on_closing)
        
    def create_ui(self):
        # Main frame layout
        main_frame = ttk.Frame(self.window)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Left frame for video
        left_frame = ttk.Frame(main_frame)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Video canvas
        self.canvas = tk.Canvas(left_frame, width=640, height=480)
        self.canvas.pack(fill=tk.BOTH, expand=True)
        
        # Right frame for data and controls
        right_frame = ttk.Frame(main_frame)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        # Stats frame
        stats_frame = ttk.LabelFrame(right_frame, text="Real-time Statistics")
        stats_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Text indicators
        self.face_status_var = tk.StringVar(value="Face: Not detected")
        self.pupil_status_var = tk.StringVar(value="Pupils: Not detected")
        self.audio_level_var = tk.StringVar(value="Audio Level: 0")
        self.stress_level_var = tk.StringVar(value="Stress Level: Low (0.0)")
        
        ttk.Label(stats_frame, textvariable=self.face_status_var).pack(anchor=tk.W)
        ttk.Label(stats_frame, textvariable=self.pupil_status_var).pack(anchor=tk.W)
        ttk.Label(stats_frame, textvariable=self.audio_level_var).pack(anchor=tk.W)
        ttk.Label(stats_frame, textvariable=self.stress_level_var).pack(anchor=tk.W)
        
        # Text sentiment input
        sentiment_frame = ttk.LabelFrame(right_frame, text="Text Sentiment Analysis")
        sentiment_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.text_input = scrolledtext.ScrolledText(sentiment_frame, wrap=tk.WORD, height=5)
        self.text_input.pack(fill=tk.X, padx=5, pady=5)
        
        analyze_btn = ttk.Button(sentiment_frame, text="Analyze Sentiment", 
                                command=self.analyze_text_sentiment)
        analyze_btn.pack(padx=5, pady=5)
        
        self.sentiment_result_var = tk.StringVar(value="Sentiment: N/A")
        ttk.Label(sentiment_frame, textvariable=self.sentiment_result_var).pack(anchor=tk.W)
        
        # Charts frame
        charts_frame = ttk.LabelFrame(right_frame, text="Monitoring Charts")
        charts_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Create matplotlib figure for charts
        self.fig = plt.Figure(figsize=(6, 8), dpi=100)
        
        # Stress level chart
        self.stress_ax = self.fig.add_subplot(311)
        self.stress_ax.set_title("Stress Level")
        self.stress_ax.set_ylim(0, 1)
        self.stress_line, = self.stress_ax.plot([], [], 'r-')
        
        # Audio level chart
        self.audio_ax = self.fig.add_subplot(312)
        self.audio_ax.set_title("Audio Level")
        self.audio_ax.set_ylim(0, 20000)
        self.audio_line, = self.audio_ax.plot([], [], 'g-')
        
        # Pupil size chart
        self.pupil_ax = self.fig.add_subplot(313)
        self.pupil_ax.set_title("Pupil Size")
        self.pupil_ax.set_ylim(0, 20)
        self.pupil_line, = self.pupil_ax.plot([], [], 'b-')
        
        self.fig.tight_layout()
        
        # Add figure to UI
        self.canvas_charts = FigureCanvasTkAgg(self.fig, charts_frame)
        self.canvas_charts.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Control buttons
        controls_frame = ttk.Frame(right_frame)
        controls_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Button(controls_frame, text="Record Voice Sample", 
                 command=self.record_audio_sample).pack(side=tk.LEFT, padx=5)
        ttk.Button(controls_frame, text="Reset Data", 
                 command=self.reset_data).pack(side=tk.LEFT, padx=5)
        ttk.Button(controls_frame, text="Exit", 
                 command=self.on_closing).pack(side=tk.RIGHT, padx=5)
    
    def process_video(self):
        if not self.running:
            return
            
        ret, frame = self.cap.read()
        
        if ret:
            # Flip horizontally for a mirror effect
            frame = cv2.flip(frame, 1)
            
            # Process the frame for face and eyes
            face, eyes, pupils = self.face_analyzer.detect_face_and_eyes(frame)
            
            # Update UI based on detection results
            if face:
                x, y, w, h = face
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                self.face_status_var.set(f"Face: Detected at ({x}, {y})")
                
                # Extract face ROI for expression analysis
                face_roi = frame[y:y+h, x:x+w]
                expression = self.face_analyzer.analyze_expression(cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY))
                
                # Draw expression on frame
                cv2.putText(frame, f"{expression['emotion']}", (x, y-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            else:
                self.face_status_var.set("Face: Not detected")
            
            # Draw eyes
            if eyes:
                for (ex, ey, ew, eh) in eyes:
                    cv2.rectangle(frame, (ex, ey), (ex+ew, ey+eh), (255, 0, 0), 2)
            
            # Draw pupils
            avg_pupil_size = 0
            if pupils:
                for (px, py, pr) in pupils:
                    cv2.circle(frame, (px, py), pr, (0, 0, 255), 2)
                    avg_pupil_size += pr
                
                avg_pupil_size /= len(pupils)
                self.pupil_status_var.set(f"Pupils: {len(pupils)} detected, avg size: {avg_pupil_size:.1f}")
                self.pupil_sizes.append(avg_pupil_size)
            else:
                self.pupil_status_var.set("Pupils: Not detected")
                self.pupil_sizes.append(0)
            
            # Get audio level
            audio_level = self.audio_recorder.get_latest_amplitude()
            self.audio_level_var.set(f"Audio Level: {audio_level:.0f}")
            self.audio_amplitudes.append(audio_level)
            
            # Calculate stress level (simple algorithm for demo)
            stress_level = self.calculate_stress_level(
                face is not None,
                avg_pupil_size,
                audio_level
            )
            
            self.stress_level_var.set(
                f"Stress Level: {self.get_stress_label(stress_level)} ({stress_level:.2f})"
            )
            self.stress_history.append(stress_level)
            
            # Update charts
            self.update_charts()
            
            # Convert frame to display in tkinter
            cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
            img = Image.fromarray(cv2image)
            imgtk = ImageTk.PhotoImage(image=img)
            
            self.canvas.create_image(0, 0, anchor=tk.NW, image=imgtk)
            self.canvas.imgtk = imgtk  # Keep a reference to prevent garbage collection
        
        # Schedule the next frame processing
        self.window.after(30, self.process_video)
    
    def get_stress_label(self, stress_level):
        if stress_level < 0.3:
            return "Low"
        elif stress_level < 0.6:
            return "Moderate"
        else:
            return "High"
    
    def calculate_stress_level(self, face_detected, pupil_size, audio_level):
        # Simple stress calculation algorithm - replace with your own model
        stress = 0.0
        
        # Face detection factor (not detecting face might indicate stress)
        if not face_detected:
            stress += 0.2
        
        # Pupil size factor (larger pupils might indicate stress)
        # Normalize pupil size to 0-1 range
        pupil_factor = min(1.0, pupil_size / 15.0) * 0.4
        stress += pupil_factor
        
        # Audio level factor (louder voice might indicate stress)
        # Normalize audio to 0-1 range
        audio_factor = min(1.0, audio_level / 10000.0) * 0.4
        stress += audio_factor
        
        return min(1.0, stress)
    
    def analyze_text_sentiment(self):
        text = self.text_input.get("1.0", tk.END).strip()
        
        if not text:
            self.sentiment_result_var.set("Sentiment: Please enter some text")
            return
            
        # Analyze sentiment
        result = analyze_sentiment(text)
        
        # Update UI
        self.sentiment_result_var.set(
            f"Sentiment: {result['label']} ({result['score']:.2f})"
        )
        
        # Add to history
        self.sentiment_history.append(result['score'])
    
    def update_charts(self):
        # Limit data points to recent history
        max_points = 100
        
        stress_data = self.stress_history[-max_points:] if self.stress_history else []
        audio_data = self.audio_amplitudes[-max_points:] if self.audio_amplitudes else []
        pupil_data = self.pupil_sizes[-max_points:] if self.pupil_sizes else []
        
        # Update data on charts
        x = range(len(stress_data))
        if stress_data:
            self.stress_line.set_data(x, stress_data)
            self.stress_ax.set_xlim(0, len(stress_data))
        
        if audio_data:
            self.audio_line.set_data(x, audio_data)
            self.audio_ax.set_xlim(0, len(audio_data))
            # Adjust y-axis based on data
            max_audio = max(audio_data) if audio_data else 10000
            self.audio_ax.set_ylim(0, max(10000, max_audio * 1.2))
        
        if pupil_data:
            self.pupil_line.set_data(x, pupil_data)
            self.pupil_ax.set_xlim(0, len(pupil_data))
        
        # Redraw
        self.canvas_charts.draw()
    
    def record_audio_sample(self):
        # Create a simple popup to show recording status
        popup = tk.Toplevel(self.window)
        popup.title("Recording")
        popup.geometry("300x100")
        
        ttk.Label(popup, text="Recording audio sample...").pack(pady=20)
        
        # Start recording in a separate thread
        def record_and_close():
            self.audio_recorder.save_audio_sample(duration=3)
            popup.destroy()
        
        threading.Thread(target=record_and_close).start()
    
    def reset_data(self):
        self.stress_history = []
        self.sentiment_history = []
        self.audio_amplitudes = []
        self.pupil_sizes = []
        
        # Reset charts
        self.stress_line.set_data([], [])
        self.audio_line.set_data([], [])
        self.pupil_line.set_data([], [])
        self.canvas_charts.draw()
    
    def on_closing(self):
        self.running = False
        if self.cap.isOpened():
            self.cap.release()
        self.audio_recorder.stop_recording()
        self.window.destroy()

# Main entry point
if __name__ == "__main__":
    root = tk.Tk()
    app = StressAnalyzerApp(root, "Visual Stress Analyzer")
    root.mainloop()