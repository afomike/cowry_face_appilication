import os
import cv2
import numpy as np
import face_recognition
from flask import Flask, render_template, request, jsonify, Response
import sqlite3
import base64
from datetime import datetime
import io
from PIL import Image
import threading
import time

app = Flask(__name__)
app.config['SECRET_KEY'] = os.urandom(24).hex()

# Global variables
camera = None
current_card_id = None
stored_face_encoding = None
verification_result = None
camera_lock = threading.Lock()

class DatabaseManager:
    def __init__(self, db_name='cowry_cards.db'):
        self.db_name = db_name
        self.init_database()
        
    def init_database(self):
        """Initialize the database with users table"""
        conn = sqlite3.connect(self.db_name)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                card_id TEXT UNIQUE NOT NULL,
                full_name TEXT NOT NULL,
                phone_number TEXT,
                email TEXT,
                passport_photo BLOB NOT NULL,
                face_encoding BLOB NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_access TIMESTAMP
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS access_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                card_id TEXT NOT NULL,
                access_granted BOOLEAN NOT NULL,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                confidence_score REAL,
                FOREIGN KEY (card_id) REFERENCES users (card_id)
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def add_sample_users(self):
        """Add sample users to the database"""
        sample_users = [
            {
                'card_id': 'COWRY001',
                'full_name': 'John Doe',
                'phone_number': '+234-801-234-5678',
                'email': 'john.doe@example.com'
            },
            {
                'card_id': 'COWRY002', 
                'full_name': 'Jane Smith',
                'phone_number': '+234-802-345-6789',
                'email': 'jane.smith@example.com'
            },
            {
                'card_id': 'COWRY003',
                'full_name': 'Ahmed Hassan',
                'phone_number': '+234-803-456-7890', 
                'email': 'ahmed.hassan@example.com'
            }
        ]
        
        for user in sample_users:
            if self.get_user_by_card_id(user['card_id']):
                continue
                
            img = Image.new('RGB', (300, 400), color=(73, 109, 137))
            img_byte_arr = io.BytesIO()
            img.save(img_byte_arr, format='JPEG')
            img_byte_arr = img_byte_arr.getvalue()
            
            blank_encoding = np.zeros(128)
            encoding_bytes = blank_encoding.tobytes()
            
            self.add_user(
                card_id=user['card_id'],
                full_name=user['full_name'],
                phone_number=user['phone_number'],
                email=user['email'],
                passport_photo=img_byte_arr,
                face_encoding=encoding_bytes
            )
    
    def add_user(self, card_id, full_name, phone_number, email, passport_photo, face_encoding):
        """Add a new user to the database"""
        conn = sqlite3.connect(self.db_name)
        cursor = conn.cursor()
        
        try:
            cursor.execute('''
                INSERT OR REPLACE INTO users (card_id, full_name, phone_number, email, passport_photo, face_encoding)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (card_id, full_name, phone_number, email, passport_photo, face_encoding))
            conn.commit()
            return True
        except Exception as e:
            print(f"Error adding user: {e}")
            return False
        finally:
            conn.close()
    
    def get_user_by_card_id(self, card_id):
        """Retrieve user information by card ID"""
        conn = sqlite3.connect(self.db_name)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT card_id, full_name, phone_number, email, passport_photo, face_encoding
            FROM users WHERE card_id = ?
        ''', (card_id,))
        
        result = cursor.fetchone()
        conn.close()
        
        if result:
            return {
                'card_id': result[0],
                'full_name': result[1], 
                'phone_number': result[2],
                'email': result[3],
                'passport_photo': result[4],
                'face_encoding': np.frombuffer(result[5], dtype=np.float64) if result[5] else None
            }
        return None
    
    def log_access_attempt(self, card_id, access_granted, confidence_score=None):
        """Log access attempt to database"""
        conn = sqlite3.connect(self.db_name)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO access_logs (card_id, access_granted, confidence_score)
            VALUES (?, ?, ?)
        ''', (card_id, access_granted, confidence_score))
        
        if access_granted:
            cursor.execute('''
                UPDATE users SET last_access = CURRENT_TIMESTAMP
                WHERE card_id = ?
            ''', (card_id,))
        
        conn.commit()
        conn.close()

class FaceRecognitionSystem:
    def __init__(self, tolerance=0.6):
        self.tolerance = tolerance
        
    def preprocess_image(self, image_data):
        """Preprocess image to ensure it's in the correct format"""
        try:
            if isinstance(image_data, bytes):
                nparr = np.frombuffer(image_data, np.uint8)
                image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            else:
                image = image_data
            
            if image is None:
                print("Error: Could not decode image")
                return None
            
            if len(image.shape) == 3 and image.shape[2] == 3:
                rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            elif len(image.shape) == 2:
                rgb_image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            else:
                rgb_image = image
            
            if rgb_image.dtype != np.uint8:
                rgb_image = rgb_image.astype(np.uint8)
            
            return rgb_image
            
        except Exception as e:
            print(f"Error preprocessing image: {e}")
            return None
        
    def extract_face_encoding(self, image_data):
        """Extract face encoding from image data"""
        try:
            rgb_image = self.preprocess_image(image_data)
            if rgb_image is None:
                return None
            
            face_locations = face_recognition.face_locations(rgb_image, model="hog")
            if not face_locations:
                print("No face locations detected")
                return None
                
            face_encodings = face_recognition.face_encodings(rgb_image, face_locations, model="small")
            if not face_encodings:
                print("No face encodings generated")
                return None
                
            return face_encodings[0]
        except Exception as e:
            print(f"Error extracting face encoding: {e}")
            return None
    
    def compare_faces(self, known_encoding, unknown_image):
        """Compare known face encoding with face from camera image"""
        try:
            unknown_encoding = self.extract_face_encoding(unknown_image)
            if unknown_encoding is None:
                return False, 0.0
            
            if isinstance(known_encoding, bytes):
                known_encoding = np.frombuffer(known_encoding, dtype=np.float64)
                        
            matches = face_recognition.compare_faces([known_encoding], unknown_encoding, tolerance=self.tolerance)
            face_distance = face_recognition.face_distance([known_encoding], unknown_encoding)[0]

            confidence = float(1 - face_distance)  # closer to 1 means better match

            # Apply BOTH tolerance and confidence threshold
            confidence_threshold = 0.55  
            is_match = bool(matches[0] and confidence >= confidence_threshold)

            return is_match, confidence

        except Exception as e:
            print(f"Error comparing faces: {e}")
            return False, 0.0

# Initialize database and face recognition system
db_manager = DatabaseManager()
face_system = FaceRecognitionSystem()

def generate_camera_frames():
    """Generate camera frames for streaming"""
    global camera, current_card_id, stored_face_encoding, verification_result
    
    while True:
        with camera_lock:
            if camera is None or not camera.isOpened():
                for camera_index in [0, 1, 2, -1]:
                    try:
                        camera = cv2.VideoCapture(camera_index)
                        if camera.isOpened():
                            camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                            camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                            camera.set(cv2.CAP_PROP_FPS, 30)
                            print(f"Camera initialized on index {camera_index}")
                            break
                    except Exception as e:
                        print(f"Failed camera index {camera_index}: {e}")
                        continue
                
                if camera is None or not camera.isOpened():
                    error_frame = np.zeros((480, 640, 3), dtype=np.uint8)
                    cv2.putText(error_frame, "CAMERA NOT AVAILABLE", (50, 240), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    ret, buffer = cv2.imencode('.jpg', error_frame)
                    if ret:
                        frame_bytes = buffer.tobytes()
                        yield (b'--frame\r\n'
                               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                    time.sleep(3)
                    continue
                
            success, frame = camera.read()
            if not success:
                print("Failed to read from camera")
                if camera:
                    camera.release()
                camera = None
                continue
            
            cv2.putText(frame, f"Card ID: {current_card_id or 'None'}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            if verification_result is not None:
                color = (0, 255, 0) if verification_result['access_granted'] else (0, 0, 255)
                status = "ACCESS GRANTED" if verification_result['access_granted'] else "ACCESS DENIED"
                cv2.putText(frame, status, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                cv2.putText(frame, f"Confidence: {verification_result['confidence']:.2f}", 
                           (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            try:
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                face_locations = face_recognition.face_locations(rgb_frame)
                for (top, right, bottom, left) in face_locations:
                    cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            except Exception as e:
                print(f"Face detection error: {e}")
            
            ret, buffer = cv2.imencode('.jpg', frame)
            if ret:
                frame_bytes = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

def add_real_user(card_id, name, phone, email, photo_path):
    """Add a real user with photo from file path"""
    print(f"Adding user: {name} with card ID: {card_id}")
    
    try:
        if not os.path.exists(photo_path):
            print(f"Error: Photo file not found at {photo_path}")
            return False
        
        with Image.open(photo_path) as pil_img:
            if pil_img.mode != 'RGB':
                pil_img = pil_img.convert('RGB')
            
            rgb_array = np.array(pil_img)
            
            if rgb_array.dtype != np.uint8:
                rgb_array = rgb_array.astype(np.uint8)
        
        face_encodings = face_recognition.face_encodings(rgb_array, model="small")
        
        if len(face_encodings) > 0:
            face_encoding = face_encodings[0]
            
            pil_image = Image.fromarray(rgb_array)
            img_byte_arr = io.BytesIO()
            pil_image.save(img_byte_arr, format='JPEG', quality=95)
            img_bytes = img_byte_arr.getvalue()
            
            success = db_manager.add_user(
                card_id=card_id,
                full_name=name,
                phone_number=phone,
                email=email,
                passport_photo=img_bytes,
                face_encoding=face_encoding.tobytes()
            )
            
            if success:
                print(f"✅ User {name} added successfully!")
                return True
            else:
                print(f"❌ Failed to add user {name} to database")
                return False
        else:
            print("❌ No face detected in image!")
            return False
            
    except Exception as e:
        print(f"❌ Error adding user: {e}")
        return False

def add_my_user():
    """Add yourself as a user - convenience function"""
    return add_real_user(
        card_id="COWRY004",
        name="Osiako Michael",
        phone="+234-815-317-8771",
        email="temidayoafote@gmail.com",
        photo_path="Photo/mike.jpg"
    )

@app.route('/')
def index():
    """Main page"""
    return render_template('index.html')

@app.route('/admin')
def admin_page():
    """Admin page for adding users"""
    return '''
    <!DOCTYPE html>
    <html>
    <head>
        <title>Cowry Card Admin - Add User</title>
        <style>
            body { font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; background: #f5f5f5; }
            .container { background: white; padding: 30px; border-radius: 10px; box-shadow: 0 5px 15px rgba(0,0,0,0.1); }
            .form-group { margin-bottom: 20px; }
            label { display: block; margin-bottom: 8px; font-weight: bold; color: #333; }
            input, select { width: 100%; padding: 12px; border: 2px solid #ddd; border-radius: 6px; font-size: 16px; }
            input:focus, select:focus { border-color: #4CAF50; outline: none; }
            button { background: #4CAF50; color: white; padding: 15px 30px; border: none; border-radius: 6px; cursor: pointer; font-size: 16px; }
            button:hover { background: #45a049; }
            #preview { max-width: 200px; margin: 10px 0; border: 3px solid #ddd; border-radius: 8px; }
            .status { padding: 15px; margin: 15px 0; border-radius: 6px; }
            .success { background: #d4edda; color: #155724; border: 1px solid #c3e6cb; }
            .error { background: #f8d7da; color: #721c24; border: 1px solid #f5c6cb; }
            .info { background: #d1ecf1; color: #0c5460; border: 1px solid #bee5eb; }
            .back-link { display: inline-block; margin-top: 20px; color: #4CAF50; text-decoration: none; }
            .back-link:hover { text-decoration: underline; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🎯 Add New Cowry Card User</h1>
            
            <form id="userForm" enctype="multipart/form-data">
                <div class="form-group">
                    <label>Card ID:</label>
                    <input type="text" id="cardId" name="cardId" required placeholder="COWRY004" style="text-transform: uppercase;">
                </div>
                
                <div class="form-group">
                    <label>Full Name:</label>
                    <input type="text" id="fullName" name="fullName" required placeholder="Your Full Name">
                </div>
                
                <div class="form-group">
                    <label>Phone Number:</label>
                    <input type="text" id="phone" name="phone" placeholder="+234-XXX-XXX-XXXX">
                </div>
                
                <div class="form-group">
                    <label>Email:</label>
                    <input type="email" id="email" name="email" placeholder="email@example.com">
                </div>
                
                <div class="form-group">
                    <label>Photo Source:</label>
                    <select id="photoSource" onchange="togglePhotoInput()">
                        <option value="file">Upload Photo File</option>
                        <option value="camera">Take Photo from Camera</option>
                    </select>
                </div>
                
                <div class="form-group" id="fileInput">
                    <label>Upload Photo (Clear frontal face photo recommended):</label>
                    <input type="file" id="photoFile" accept="image/*" onchange="previewImage()">
                    <img id="preview" style="display:none;">
                </div>
                
                <div class="form-group" id="cameraInput" style="display:none;">
                    <label>Camera Photo:</label>
                    <video id="video" width="400" height="300" autoplay style="border: 2px solid #ddd; border-radius: 6px;"></video><br><br>
                    <button type="button" onclick="capturePhoto()">📸 Capture Photo</button>
                    <canvas id="canvas" width="400" height="300" style="display:none;"></canvas>
                </div>
                
                <button type="submit">✅ Add User</button>
            </form>
            
            <div id="status"></div>
            
            <a href="/" class="back-link">← Back to Main System</a>
        </div>
        
        <script>
            let stream = null;
            let capturedPhotoData = null;
            
            function togglePhotoInput() {
                const source = document.getElementById('photoSource').value;
                const fileInput = document.getElementById('fileInput');
                const cameraInput = document.getElementById('cameraInput');
                
                if (source === 'camera') {
                    fileInput.style.display = 'none';
                    cameraInput.style.display = 'block';
                    startCamera();
                } else {
                    fileInput.style.display = 'block';
                    cameraInput.style.display = 'none';
                    stopCamera();
                }
            }
            
            async function startCamera() {
                try {
                    stream = await navigator.mediaDevices.getUserMedia({ 
                        video: { width: 400, height: 300 } 
                    });
                    document.getElementById('video').srcObject = stream;
                } catch (err) {
                    showStatus('Camera access denied or not available', 'error');
                }
            }
            
            function stopCamera() {
                if (stream) {
                    stream.getTracks().forEach(track => track.stop());
                    stream = null;
                }
            }
            
            function capturePhoto() {
                const video = document.getElementById('video');
                const canvas = document.getElementById('canvas');
                const ctx = canvas.getContext('2d');
                
                ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
                capturedPhotoData = canvas.toDataURL('image/jpeg', 0.95);
                
                const preview = document.getElementById('preview');
                preview.src = capturedPhotoData;
                preview.style.display = 'block';
                
                showStatus('Photo captured successfully! Good lighting and clear face are important.', 'success');
            }
            
            function previewImage() {
                const file = document.getElementById('photoFile').files[0];
                if (file) {
                    const reader = new FileReader();
                    reader.onload = function(e) {
                        const preview = document.getElementById('preview');
                        preview.src = e.target.result;
                        preview.style.display = 'block';
                    };
                    reader.readAsDataURL(file);
                }
            }
            
            function showStatus(message, type) {
                const status = document.getElementById('status');
                status.textContent = message;
                status.className = `status ${type}`;
            }
            
            document.getElementById('userForm').addEventListener('submit', async function(e) {
                e.preventDefault();
                
                const photoSource = document.getElementById('photoSource').value;
                
                let photoData = null;
                if (photoSource === 'camera') {
                    if (!capturedPhotoData) {
                        showStatus('Please capture a photo first!', 'error');
                        return;
                    }
                    photoData = capturedPhotoData.split(',')[1];
                } else {
                    const fileInput = document.getElementById('photoFile');
                    if (!fileInput.files[0]) {
                        showStatus('Please select a photo file!', 'error');
                        return;
                    }
                    const reader = new FileReader();
                    photoData = await new Promise((resolve) => {
                        reader.onload = () => resolve(reader.result.split(',')[1]);
                        reader.readAsDataURL(fileInput.files[0]);
                    });
                }
                
                const userData = {
                    card_id: document.getElementById('cardId').value.toUpperCase(),
                    full_name: document.getElementById('fullName').value,
                    phone_number: document.getElementById('phone').value,
                    email: document.getElementById('email').value,
                    passport_photo: photoData
                };
                
                try {
                    showStatus('Adding user... Please wait, this may take a few moments.', 'info');
                    
                    const response = await fetch('/add_user', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify(userData)
                    });
                    
                    const result = await response.json();
                    
                    if (response.ok) {
                        showStatus(`✅ User ${userData.full_name} added successfully! You can now test with card ID: ${userData.card_id}`, 'success');
                        document.getElementById('userForm').reset();
                        document.getElementById('preview').style.display = 'none';
                        capturedPhotoData = null;
                    } else {
                        showStatus('❌ ' + result.error, 'error');
                    }
                } catch (error) {
                    showStatus('❌ Error adding user: ' + error.message, 'error');
                }
            });
            
            document.getElementById('cardId').addEventListener('input', function(e) {
                e.target.value = e.target.value.toUpperCase();
            });
        </script>
    </body>
    </html>
    '''

@app.route('/video_feed')
def video_feed():
    """Video streaming route"""
    return Response(generate_camera_frames(),
                   mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/scan_card', methods=['POST'])
def scan_card():
    """Handle card scanning"""
    global current_card_id, stored_face_encoding, verification_result
    
    data = request.json
    card_id = data.get('card_id', '').strip().upper()
    
    if not card_id:
        return jsonify({'error': 'Card ID is required'}), 400
    
    user = db_manager.get_user_by_card_id(card_id)
    if not user:
        return jsonify({'error': 'Card not found in system'}), 404
    
    current_card_id = card_id
    verification_result = None
    
    passport_photo_b64 = base64.b64encode(user['passport_photo']).decode('utf-8')
    
    stored_face_encoding = user['face_encoding']
    
    return jsonify({
        'success': True,
        'user': {
            'card_id': user['card_id'],
            'full_name': user['full_name'],
            'phone_number': user['phone_number'],
            'email': user['email'],
            'passport_photo': passport_photo_b64
        }
    })

@app.route('/verify_face', methods=['POST'])
def verify_face():
    """Perform face verification"""
    global current_card_id, stored_face_encoding, verification_result
    
    if not current_card_id or stored_face_encoding is None:
        return jsonify({'error': 'No card scanned or user data not loaded'}), 400
    
    with camera_lock:
        if camera is None:
            return jsonify({'error': 'Camera not initialized'}), 500
            
        ret, frame = camera.read()
        if not ret:
            return jsonify({'error': 'Failed to capture image from camera'}), 500
    
    ret, buffer = cv2.imencode('.jpg', frame)
    if not ret:
        return jsonify({'error': 'Failed to process camera image'}), 500
    
    frame_bytes = buffer.tobytes()
    
    is_match, confidence = face_system.compare_faces(stored_face_encoding, frame_bytes)
    
    # Ensure values are JSON serializable
    is_match = bool(is_match)
    confidence = float(confidence)
    
    db_manager.log_access_attempt(current_card_id, is_match, confidence)
    
    verification_result = {
        'access_granted': is_match,
        'confidence': confidence,
        'timestamp': datetime.now().isoformat()
    }
    
    def clear_verification():
        time.sleep(5)
        global verification_result
        verification_result = None
    
    threading.Thread(target=clear_verification, daemon=True).start()

    
    return jsonify({
        'access_granted': is_match,
        'confidence': confidence,
        'message': 'Access granted - Face verified!' if is_match else 'Access denied - Face not recognized',
        'timestamp': datetime.now().isoformat()
    })

@app.route('/reset_session', methods=['POST'])
def reset_session():
    """Reset the current session"""
    global current_card_id, stored_face_encoding, verification_result
    
    current_card_id = None
    stored_face_encoding = None
    verification_result = None
    
    return jsonify({'success': True, 'message': 'Session reset'})

@app.route('/add_user', methods=['POST'])
def add_user():
    """Add a new user via web interface"""
    try:
        data = request.json
        
        photo_data = base64.b64decode(data['passport_photo'])
        face_encoding = face_system.extract_face_encoding(photo_data)
        
        if face_encoding is None:
            return jsonify({'error': 'No face detected in uploaded photo. Please ensure the photo shows a clear frontal face with good lighting.'}), 400
        
        encoding_bytes = face_encoding.tobytes()
        
        success = db_manager.add_user(
            card_id=data['card_id'].upper(),
            full_name=data['full_name'],
            phone_number=data.get('phone_number', ''),
            email=data.get('email', ''),
            passport_photo=photo_data,
            face_encoding=encoding_bytes
        )
        
        if success:
            return jsonify({'success': True, 'message': f'User {data["full_name"]} added successfully with card ID: {data["card_id"].upper()}'})
        else:
            return jsonify({'error': 'Failed to add user to database'}), 400
            
    except Exception as e:
        print(f"Error in add_user: {e}")
        return jsonify({'error': f'Error processing request: {str(e)}'}), 500

@app.route('/check_camera', methods=['GET'])
def check_camera():
    """Check camera availability"""
    try:
        test_camera = cv2.VideoCapture(0)
        if test_camera.isOpened():
            ret, frame = test_camera.read()
            test_camera.release()
            if ret:
                return jsonify({'camera_available': True, 'message': 'Camera is working'})
        
        return jsonify({'camera_available': False, 'message': 'Camera not available'})
    except Exception as e:
        return jsonify({'camera_available': False, 'message': f'Camera error: {str(e)}'})

@app.route('/restart_camera', methods=['POST'])
def restart_camera():
    """Restart camera connection"""
    global camera
    
    try:
        with camera_lock:
            if camera:
                camera.release()
            camera = None
            
        return jsonify({'success': True, 'message': 'Camera restarted'})
    except Exception as e:
        return jsonify({'success': False, 'message': f'Error restarting camera: {str(e)}'})

@app.route('/list_users', methods=['GET'])
def list_users():
    """List all registered users (admin function)"""
    try:
        conn = sqlite3.connect(db_manager.db_name)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT card_id, full_name, phone_number, email, created_at, last_access
            FROM users ORDER BY created_at DESC
        ''')
        
        users = []
        for row in cursor.fetchall():
            users.append({
                'card_id': row[0],
                'full_name': row[1],
                'phone_number': row[2],
                'email': row[3],
                'created_at': row[4],
                'last_access': row[5]
            })
        
        conn.close()
        return jsonify({'users': users})
        
    except Exception as e:
        return jsonify({'error': f'Error retrieving users: {str(e)}'}), 500

@app.route('/access_logs', methods=['GET'])
def access_logs():
    """Get access logs (admin function)"""
    try:
        conn = sqlite3.connect(db_manager.db_name)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT al.card_id, u.full_name, al.access_granted, al.confidence_score, al.timestamp
            FROM access_logs al
            LEFT JOIN users u ON al.card_id = u.card_id
            ORDER BY al.timestamp DESC
            LIMIT 100
        ''')
        
        logs = []
        for row in cursor.fetchall():
            logs.append({
                'card_id': row[0],
                'full_name': row[1] or 'Unknown',
                'access_granted': bool(row[2]),
                'confidence_score': row[3],
                'timestamp': row[4]
            })
        
        conn.close()
        return jsonify({'logs': logs})
        
    except Exception as e:
        return jsonify({'error': f'Error retrieving logs: {str(e)}'}), 500

if __name__ == '__main__':
    print("🔧 Initializing Cowry Card System...")
    print("=" * 50)
    
    try:
        print("📝 Adding sample users to database...")
        db_manager.add_sample_users()
        print("✅ Sample users added (COWRY001, COWRY002, COWRY003)")
        
        print("👤 Checking for real user photo...")
        if add_my_user():
            print("✅ Real user added successfully!")
        else:
            print("⚠️  Real user not added - photo may not exist")
            print("   You can add users through the web interface at /admin")
        
        print("=" * 50)
        print("🚀 Starting Flask server...")
        print("📱 Main interface: http://localhost:5000")
        print("👨‍💼 Admin interface: http://localhost:5000/admin")
        print("=" * 50)
        
        app.run(host='0.0.0.0', port=5000, debug=True, threaded=True)
        
    except Exception as e:
        print(f"❌ Error starting application: {e}")
        print("Check that all dependencies are installed:")
        print("pip install flask opencv-python face_recognition numpy pillow")
    finally:
        if camera:
            camera.release()