from flask import Flask, request, render_template, jsonify
import cv2
import mediapipe as mp
import numpy as np
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB max upload
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose(static_image_mode=False, model_complexity=1, enable_segmentation=False, min_detection_confidence=0.5)

def process_video(video_path):
    """
    Process uploaded video: Detect poses and analyze squat form.
    Returns a list of frame analyses (e.g., angles, feedback).
    """
    cap = cv2.VideoCapture(video_path)
    analyses = []
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Flip frame horizontally for selfie-view
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Pose detection
        results = pose.process(rgb_frame)
        
        if results.pose_landmarks:
            # Draw landmarks (optional, for debugging)
            mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            
            # Extract keypoints (normalized landmarks)
            landmarks = results.pose_landmarks.landmark
            
            # Key points for squat analysis (indices from MediaPipe)
            left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
            left_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x, landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
            left_knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
            left_ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
            
            # Calculate angles (in degrees)
            hip_angle = calculate_angle(left_shoulder, left_hip, left_knee)
            knee_angle = calculate_angle(left_hip, left_knee, left_ankle)
            
            # Simple squat form checks (thresholds based on ideal form)
            feedback = []
            if knee_angle < 80:  # Too shallow squat
                feedback.append("Squat deeper! Knee angle should be around 90 degrees.")
            if abs(hip_angle - 180) > 20:  # Back not straight (hip angle deviation)
                feedback.append("Keep your back straight! Avoid arching.")
            if left_knee[0] > left_ankle[0] + 0.05:  # Knee past toes (x-coordinate check)
                feedback.append("Knees shouldn't go past toes. Adjust stance.")
            
            if not feedback:
                feedback.append("Good form! Keep it up.")
            
            analyses.append({
                'frame_num': len(analyses),
                'hip_angle': round(hip_angle, 2),
                'knee_angle': round(knee_angle, 2),
                'feedback': feedback[0]  # Pick main feedback per frame
            })
        
        # Optional: Save processed frame (uncomment for debugging)
        # cv2.imwrite(f'output/frame_{len(analyses)}.jpg', frame)
    
    cap.release()
    return analyses

def calculate_angle(a, b, c):
    """
    Calculate angle at point b between points a, b, c.
    Uses law of cosines.
    """
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    if angle > 180.0:
        angle = 360 - angle
    return angle

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_video():
    if 'video' not in request.files:
        return jsonify({'error': 'No video uploaded'}), 400
    
    file = request.files['video']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if file:
        filename = secure_filename(file.filename)
        video_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(video_path)
        
        try:
            analyses = process_video(video_path)
            # Summarize: Average angles and common feedback
            avg_hip = np.mean([a['hip_angle'] for a in analyses])
            avg_knee = np.mean([a['knee_angle'] for a in analyses])
            feedbacks = [a['feedback'] for a in analyses]
            summary_feedback = max(set(feedbacks), key=feedbacks.count)  # Most common feedback
            
            os.remove(video_path)  # Clean up
            return jsonify({
                'success': True,
                'avg_hip_angle': round(avg_hip, 2),
                'avg_knee_angle': round(avg_knee, 2),
                'feedback': summary_feedback,
                'total_frames_analyzed': len(analyses)
            })
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    return jsonify({'error': 'Upload failed'}), 400

if __name__ == '__main__':
    app.run(debug=True)
    