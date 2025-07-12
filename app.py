from flask import Flask, render_template, Response, request, redirect, url_for, flash, jsonify, session
import cv2
import os
import numpy as np
import mediapipe as mp
from keras_facenet import FaceNet
import pickle
from datetime import datetime, timedelta
import time
import ssl
from dotenv import load_dotenv
load_dotenv()
import threading
from queue import Queue
from twilio.rest import Client
from email.message import EmailMessage
import smtplib
from playsound import playsound
from werkzeug.security import generate_password_hash, check_password_hash
import secrets
import base64
from sqlalchemy import create_engine, Column, Integer, String, DateTime, Text, ForeignKey, func
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, scoped_session
from io import BytesIO
from PIL import Image

# Initialize queue for background processing
task_queue = Queue()

# Flask application setup
app = Flask(__name__)
app.secret_key = secrets.token_hex(24)
app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(hours=2)

# SQLite Database Configuration
BASE_PATH = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_PATH, "database.db")
app.config['SQLALCHEMY_DATABASE_URI'] = f'sqlite:///{DB_PATH}'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Create the SQLAlchemy engine
engine = create_engine(app.config['SQLALCHEMY_DATABASE_URI'])
Base = declarative_base()

# Create session factory
session_factory = sessionmaker(bind=engine)
db_session = scoped_session(session_factory)

# Division access code for registration
DIVISION_ACCESS_CODE = os.getenv("DIVISION_ACCESS_CODE", "POLICE2024")

# Email Configuration
EMAIL_SENDER = os.getenv("EMAIL_ADDRESS")
EMAIL_PASSWORD = os.getenv("EMAIL_PASSWORD")
EMAIL_RECEIVER = os.getenv("RECIPIENT_EMAIL")

# Twilio Configuration
TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN")
TWILIO_PHONE_NUMBER = os.getenv("TWILIO_FROM_NUMBER")
RECIPIENT_PHONE_NUMBER = os.getenv("TWILIO_TO_NUMBER")

# Directories
DETECTED_FACES_FOLDER = os.path.join(BASE_PATH, "detected_faces")
TIME_DATA_FOLDER = os.path.join(BASE_PATH, "time_data")
EMBEDDINGS_FILE = os.path.join(BASE_PATH, "embeddings.pkl")
UPLOAD_FOLDER = os.path.join(BASE_PATH, "uploads")
DATASET_FOLDER = os.path.join(BASE_PATH, "dataset")

os.makedirs(DETECTED_FACES_FOLDER, exist_ok=True)
os.makedirs(TIME_DATA_FOLDER, exist_ok=True)
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(DATASET_FOLDER, exist_ok=True)

# SQLAlchemy Models
class Officer(Base):
    __tablename__ = 'officers'
    
    id = Column(Integer, primary_key=True)
    badge_number = Column(String(10), unique=True, nullable=False)
    full_name = Column(String(100), nullable=False)
    rank = Column(String(50), nullable=False)
    email = Column(String(100), unique=True, nullable=False)
    password_hash = Column(String(255), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

class Suspect(Base):
    __tablename__ = 'suspects'
    
    id = Column(Integer, primary_key=True)
    suspect_name = Column(String(100), nullable=False)
    image_path = Column(String(255), nullable=False)
    detected_time = Column(DateTime, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

# Load Face Recognition Model
embedder = FaceNet()
mp_face_detection = mp.solutions.face_detection
face_detector = mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)

# Load known faces embeddings
try:
    with open(EMBEDDINGS_FILE, "rb") as f:
        known_faces = pickle.load(f)
except (FileNotFoundError, EOFError):
    known_faces = {}

# Initialize database
def init_db():
    try:
        Base.metadata.create_all(engine)
        print("Database initialized successfully")
    except Exception as e:
        print(f"Error initializing database: {e}")

# Face recognition functions
def extract_face(img):
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = face_detector.process(img_rgb)
    faces = []

    if results.detections:
        for detection in results.detections:
            bboxC = detection.location_data.relative_bounding_box
            h, w, _ = img.shape
            x, y, width, height = (int(bboxC.xmin * w), int(bboxC.ymin * h), 
                                  int(bboxC.width * w), int(bboxC.height * h))
            face = img_rgb[y:y + height, x:x + width]
            if face.shape[0] > 0 and face.shape[1] > 0:
                faces.append((face, (x, y, width, height)))
    return faces

def recognize_face(face_embedding):
    face_embedding = face_embedding / np.linalg.norm(face_embedding)
    min_dist = float("inf")
    name = "Unknown"

    for person, embeddings in known_faces.items():
        for saved_embedding in embeddings:
            dist = np.linalg.norm(face_embedding - saved_embedding)
            if dist < 0.7 and dist < min_dist:
                min_dist = dist
                name = person
    return name

# Alert mechanisms
def make_alert_call(name, timestamp):
    """Make a phone call alert when a suspect is detected"""
    try:
        client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
        
        # Create a TwiML response with text-to-speech
        twiml = f"""
        <Response>
            <Say>Alert! Suspect {name} has been detected at {timestamp}. Please check your email for more details.</Say>
            <Pause length="1"/>
            <Say>Repeating: Suspect {name} has been detected.</Say>
        </Response>
        """
        
        # Make the call
        call = client.calls.create(
            twiml=twiml,
            to=RECIPIENT_PHONE_NUMBER,
            from_=TWILIO_PHONE_NUMBER
        )
        
        print(f"Phone alert initiated for suspect: {name}, Call SID: {call.sid}")
        return True
    except Exception as e:
        print(f"Error making phone call: {e}")
        return False

def send_email_alert(name, timestamp, face_path):
    if not EMAIL_SENDER or not EMAIL_PASSWORD or not EMAIL_RECEIVER:
        print("Email configuration not set up")
        return

    subject = f"Suspect Detected: {name}"
    body = f"A suspect has been detected!!!!.\n\nName: {name}\nTime: {timestamp}"

    msg = EmailMessage()
    msg['From'] = EMAIL_SENDER
    msg['To'] = EMAIL_RECEIVER
    msg['Subject'] = subject
    msg.set_content(body)

    # Attach the detected face image
    try:
        with open(face_path, 'rb') as img:
            img_data = img.read()
            img_name = os.path.basename(face_path)
            msg.add_attachment(img_data, maintype='image', subtype='jpeg', filename=img_name)
    except Exception as e:
        print(f"Error attaching image: {e}")

    context = ssl.create_default_context()
    try:
        with smtplib.SMTP_SSL('smtp.gmail.com', 465, context=context) as smtp:
            smtp.login(EMAIL_SENDER, EMAIL_PASSWORD)
            smtp.sendmail(EMAIL_SENDER, EMAIL_RECEIVER, msg.as_string())
            print(f"Email alert sent for suspect: {name}")
    except Exception as e:
        print(f"Email sending failed: {e}")

def process_alerts():
    """Thread to process suspect alerts asynchronously"""
    while True:
        task = task_queue.get()
        if task is None:
            break  # Stop the thread when None is added to the queue

        name, timestamp, face_path = task
        try:
            # Save suspect in SQLite database
            session = db_session()
            suspect = Suspect(
                suspect_name=name, 
                image_path=face_path, 
                detected_time=datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S")
            )
            session.add(suspect)
            session.commit()
            session.close()

            # Send Email (if configured)
            if EMAIL_SENDER and EMAIL_PASSWORD and EMAIL_RECEIVER:
                send_email_alert(name, timestamp, face_path)
            
            # Make phone call alert (if configured)
            if TWILIO_ACCOUNT_SID and TWILIO_AUTH_TOKEN and TWILIO_PHONE_NUMBER:
                make_alert_call(name, timestamp)

        except Exception as e:
            print(f"Error processing alert: {e}")

        task_queue.task_done()

# Start the background thread
alert_thread = threading.Thread(target=process_alerts, daemon=True)
alert_thread.start()

# Video processing functions
def extract_faces_from_video(name, video_path, num_images=500):
    if not os.path.exists(video_path):
        return "Error: Video file does not exist."

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return "Error: Unable to open video file."

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(1, total_frames // num_images)

    output_dir = os.path.join(DATASET_FOLDER, name)
    os.makedirs(output_dir, exist_ok=True)

    frame_count = 0
    saved_images = 0
    new_embeddings = []

    while saved_images < num_images:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % step != 0:
            frame_count += 1
            continue

        frame_count += 1
        faces = extract_face(frame)

        for face, _ in faces:
            face_resized = cv2.resize(face, (160, 160))
            embedding = embedder.embeddings([face_resized])[0]
            new_embeddings.append(embedding)

            image_path = os.path.join(output_dir, f"face_{saved_images}.jpg")
            cv2.imwrite(image_path, cv2.cvtColor(face_resized, cv2.COLOR_RGB2BGR))
            saved_images += 1

    cap.release()

    # Save embeddings to embeddings.pkl
    if new_embeddings:
        if os.path.exists(EMBEDDINGS_FILE):
            with open(EMBEDDINGS_FILE, "rb") as f:
                current_known_faces = pickle.load(f)
        else:
            current_known_faces = {}

        if name in current_known_faces:
            current_known_faces[name].extend(new_embeddings)
        else:
            current_known_faces[name] = new_embeddings

        with open(EMBEDDINGS_FILE, "wb") as f:
            pickle.dump(current_known_faces, f)

        # Update global known_faces
        global known_faces
        known_faces = current_known_faces

        return f"Extracted {len(new_embeddings)} face embeddings for {name} and saved in {EMBEDDINGS_FILE}."
    
    return "No faces detected in the video."

alerts = []
last_detection_time = {}

# NEW: Process frame from client-side camera
def process_frame(frame_data):
    """Process a single frame from client-side camera"""
    global alerts, last_detection_time
    
    try:
        # Decode base64 image
        image_data = base64.b64decode(frame_data.split(',')[1])
        image = Image.open(BytesIO(image_data))
        frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        # Reload known faces dynamically
        if os.path.exists(EMBEDDINGS_FILE):
            with open(EMBEDDINGS_FILE, "rb") as f:
                current_known_faces = pickle.load(f)
                global known_faces
                known_faces = current_known_faces

        detections = []
        faces = extract_face(frame)
        
        for face, (x, y, width, height) in faces:
            face_resized = cv2.resize(face, (160, 160))
            face_embedding = embedder.embeddings([face_resized])[0]
            name = recognize_face(face_embedding)

            detections.append({
                'name': name,
                'bbox': {'x': x, 'y': y, 'width': width, 'height': height}
            })

            current_time = time.time()

            if name != "Unknown" and (name not in last_detection_time or current_time - last_detection_time[name] >= 10):
                last_detection_time[name] = current_time  

                person_folder = os.path.join(DETECTED_FACES_FOLDER, name)
                os.makedirs(person_folder, exist_ok=True)

                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                face_count = len(os.listdir(person_folder))
                face_filename = f"img_{face_count + 1}.jpg"
                face_path = os.path.join(person_folder, face_filename)

                cv2.imwrite(face_path, cv2.cvtColor(face, cv2.COLOR_RGB2BGR))

                time_data_path = os.path.join(TIME_DATA_FOLDER, f"{name}.txt")
                with open(time_data_path, "a") as f:
                    f.write(f"{timestamp}\n")

                alerts.append({"name": name, "time": timestamp})

                # Add task to the queue instead of blocking frame processing
                task_queue.put((name, timestamp, face_path))

        return detections
        
    except Exception as e:
        print(f"Error processing frame: {e}")
        return []

# REMOVED: The old generate_frames function that used cv2.VideoCapture(0)

# Routes for authentication
@app.route('/')
def login():
    if 'badge_number' in session:
        return redirect(url_for('landing'))
    return render_template('login.html')

@app.route('/login', methods=['POST'])
def login_post():
    badge_number = request.form.get('badge_number')
    password = request.form.get('password')
    
    if not badge_number or not password:
        flash('Please provide both badge number and password', 'error')
        return redirect(url_for('login'))
    
    try:
        db = db_session()
        officer = db.query(Officer).filter_by(badge_number=badge_number).first()
        
        if officer and check_password_hash(officer.password_hash, password):
            session.permanent = True
            session['badge_number'] = badge_number
            session['full_name'] = officer.full_name
            session['rank'] = officer.rank
            flash(f'Welcome, {officer.rank} {officer.full_name}!', 'success')
            return redirect(url_for('landing'))
        else:
            flash('Invalid credentials. Please try again.', 'error')
    except Exception as e:
        flash(f'Error during login: {str(e)}', 'error')
    finally:
        db.close()
    
    return redirect(url_for('login'))

@app.route('/register')
def register():
    return render_template('registration.html')

@app.route('/register', methods=['POST'])
def register_post():
    badge_number = request.form.get('badge_number')
    full_name = request.form.get('full_name')
    rank = request.form.get('rank')
    email = request.form.get('email')
    password = request.form.get('password')
    confirm_password = request.form.get('confirm_password')
    division_code = request.form.get('division_code')
    
    # Validate input
    if not all([badge_number, full_name, rank, email, password, confirm_password, division_code]):
        flash('All fields are required', 'error')
        return redirect(url_for('register'))
    
    if password != confirm_password:
        flash('Passwords do not match', 'error')
        return redirect(url_for('register'))
    
    if division_code != DIVISION_ACCESS_CODE:
        flash('Invalid division access code', 'error')
        return redirect(url_for('register'))
    
    try:
        db = db_session()
        
        # Check if badge number or email already exists
        existing_user = db.query(Officer).filter(
            (Officer.badge_number == badge_number) | (Officer.email == email)
        ).first()
        
        if existing_user:
            flash('Badge number or email already registered', 'error')
            return redirect(url_for('register'))
        
        # Create new officer
        password_hash = generate_password_hash(password)
        new_officer = Officer(
            badge_number=badge_number,
            full_name=full_name,
            rank=rank,
            email=email,
            password_hash=password_hash
        )
        db.add(new_officer)
        db.commit()
        
        flash('Registration successful! You can now log in', 'success')
        return redirect(url_for('login'))
        
    except Exception as e:
        flash(f'Error during registration: {str(e)}', 'error')
        return redirect(url_for('register'))
    finally:
        db.close()

@app.route('/logout')
def logout():
    session.clear()
    flash('You have been logged out', 'info')
    return redirect(url_for('login'))

# Routes for application features
@app.route('/landing')
def landing():
    if 'badge_number' not in session:
        return redirect(url_for('login'))
    return render_template('landing.html')

@app.route('/get_alerts')
def get_alerts():
    global alerts
    return jsonify(alerts)

@app.route('/suspects')
def get_suspects():
    if 'badge_number' not in session:
        return redirect(url_for('login'))
        
    try:
        db = db_session()
        suspects = db.query(Suspect).order_by(Suspect.detected_time.desc()).all()
        
        # Add base64 image data for each suspect
        suspects_with_images = []
        for suspect in suspects:
            suspect_dict = {
                'id': suspect.id,
                'suspect_name': suspect.suspect_name,
                'image_path': suspect.image_path,
                'detected_time': suspect.detected_time,
                'created_at': suspect.created_at
            }
            
            try:
                with open(suspect.image_path, 'rb') as img_file:
                    img_data = img_file.read()
                    suspect_dict['image_b64'] = base64.b64encode(img_data).decode('utf-8')
            except Exception as e:
                print(f"Error reading image file: {e}")
                suspect_dict['image_b64'] = ''
                
            suspects_with_images.append(suspect_dict)
        
        return render_template('suspects.html', suspects=suspects_with_images)
    except Exception as e:
        flash(f'Error retrieving suspects: {str(e)}', 'error')
        return redirect(url_for('landing'))
    finally:
        db.close()

@app.route('/live-mon')
def live_mon():
    if 'badge_number' not in session:
        return redirect(url_for('login'))
    global alerts
    alerts = []  # Clear previous alerts
    return render_template("livemon.html")

# NEW: Route to process frames from client-side camera
@app.route('/process_frame', methods=['POST'])
def process_frame_route():
    if 'badge_number' not in session:
        return jsonify({'error': 'Not authenticated'}), 401
        
    try:
        data = request.get_json()
        frame_data = data.get('frame')
        
        if not frame_data:
            return jsonify({'error': 'No frame data provided'}), 400
            
        detections = process_frame(frame_data)
        return jsonify({'detections': detections})
        
    except Exception as e:
        print(f"Error in process_frame_route: {e}")
        return jsonify({'error': 'Frame processing failed'}), 500

# REMOVED: /video_feed route since we're not using server-side camera

@app.route('/delete_face', methods=['GET', 'POST'])
def delete_face():
    if 'badge_number' not in session:
        return redirect(url_for('login'))
        
    embeddings_path = EMBEDDINGS_FILE
    
    # Load existing data
    try:
        with open(embeddings_path, 'rb') as f:
            embeddings = pickle.load(f)
    except Exception as e:
        print("⚠️ Error loading embeddings:", e)
        embeddings = {}

    if request.method == 'POST':
        name_to_delete = request.form['name']
        if name_to_delete in embeddings:
            del embeddings[name_to_delete]
            with open(embeddings_path, 'wb') as f:
                pickle.dump(embeddings, f)
            # Update global known_faces
            global known_faces
            known_faces = embeddings
            message = f"✅ Deleted: {name_to_delete}"
        else:
            message = f"❌ Name '{name_to_delete}' not found."
        return render_template("delete_face.html", names=list(embeddings.keys()), message=message)

    return render_template("delete_face.html", names=list(embeddings.keys()), message=None)

@app.route('/contact')
def contact():
    if 'badge_number' not in session:
        return redirect(url_for('login'))
    return render_template('contact.html')

@app.route('/new-crim', methods=['GET', 'POST'])
def newcrim():
    if 'badge_number' not in session:
        return redirect(url_for('login'))
        
    if request.method == 'POST':
        name = request.form.get("name")
        file = request.files["video"]

        if file and name:
            filepath = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(filepath)

            result = extract_faces_from_video(name, filepath)
            flash(result)
            return redirect(url_for('newcrim'))

    return render_template('newcrim.html')

# Cleanup function for the application
@app.teardown_appcontext
def shutdown_session(exception=None):
    db_session.remove()

if __name__ == "__main__":
    init_db()  # Initialize database tables on startup
    app.run(host="0.0.0.0", debug=True)