from flask import Flask, render_template, Response, request, redirect, url_for, flash, jsonify
import cv2
import os
import numpy as np
import mediapipe as mp
from keras_facenet import FaceNet
import pickle
from datetime import datetime
import time
from pymongo import MongoClient
from email.message import EmailMessage
import smtplib
import ssl
import threading
from queue import Queue
from twilio.rest import Client

task_queue = Queue()

app = Flask(__name__)
app.secret_key = "your_secret_key"

# Email Configuration
EMAIL_SENDER = 'mourishantonyc@gmail.com'
EMAIL_PASSWORD = 'hmxn wppp myla mhkc'
EMAIL_RECEIVER = 'rajmourishantony@gmail.com'

# Twilio Configuration
TWILIO_ACCOUNT_SID = 'ACb12d8abd2288f3d831aa9003bc7ff69e'
TWILIO_AUTH_TOKEN = '8b9019e92d1bb035b14bc9943d06a939'    
TWILIO_PHONE_NUMBER = '+18596591506'      
RECIPIENT_PHONE_NUMBER = '+916381032833' 

# Connect to MongoDB
client = MongoClient("mongodb://localhost:27017/")
db = client["face_recognition"]
collection = db["suspects"]

# Load Face Recognition Model
embedder = FaceNet()
mp_face_detection = mp.solutions.face_detection
face_detector = mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)

# Directories
BASE_PATH = os.path.dirname(os.path.abspath(__file__))
DETECTED_FACES_FOLDER = os.path.join(BASE_PATH, "detected_faces")
TIME_DATA_FOLDER = os.path.join(BASE_PATH, "time_data")
EMBEDDINGS_FILE = os.path.join(BASE_PATH, "embeddings.pkl")
UPLOAD_FOLDER = os.path.join(BASE_PATH, "uploads")
DATASET_FOLDER = os.path.join(BASE_PATH, "dataset")

os.makedirs(DETECTED_FACES_FOLDER, exist_ok=True)
os.makedirs(TIME_DATA_FOLDER, exist_ok=True)
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(DATASET_FOLDER, exist_ok=True)

# Load known faces embeddings
with open(EMBEDDINGS_FILE, "rb") as f:
    known_faces = pickle.load(f)


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

alerts = []
last_detection_time = {}

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
                known_faces = pickle.load(f)
        else:
            known_faces = {}

        if name in known_faces:
            known_faces[name].extend(new_embeddings)
        else:
            known_faces[name] = new_embeddings

        with open(EMBEDDINGS_FILE, "wb") as f:
            pickle.dump(known_faces, f)

        return f"Extracted {len(new_embeddings)} face embeddings for {name} and saved in {EMBEDDINGS_FILE}."
    
    return "No faces detected in the video."

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
    subject = f"Suspect Detected  : {name}"
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

# Update in generate_frames() to send email alerts
from playsound import playsound

def process_alerts():
    """Thread to process suspect alerts asynchronously"""
    while True:
        task = task_queue.get()
        if task is None:
            break  # Stop the thread when None is added to the queue

        name, timestamp, face_path = task
        try:
            # Save suspect in MongoDB
            with open(face_path, "rb") as img_file:
                image_data = img_file.read()

            suspect_data = {
                "suspect_name": name,
                "detected_image": image_data,
                "time": timestamp
            }
            collection.insert_one(suspect_data)

            # Play alert sound
            alert_sound_path = os.path.join(BASE_PATH, "static", "alert.mp3")
            playsound(alert_sound_path)

            # Send Email
            send_email_alert(name, timestamp, face_path)
            
            # Make phone call alert
            make_alert_call(name, timestamp)

        except Exception as e:
            print(f"Error processing alert: {e}")

        task_queue.task_done()

# Start the background thread
alert_thread = threading.Thread(target=process_alerts, daemon=True)
alert_thread.start()


def generate_frames():
    global alerts, last_detection_time
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        return

    while True:
        success, frame = cap.read()
        if not success:
            break
        frame = cv2.flip(frame, 1)

        # Reload known faces dynamically
        if os.path.exists(EMBEDDINGS_FILE):
         with open(EMBEDDINGS_FILE, "rb") as f:
            known_faces = pickle.load(f)

        faces = extract_face(frame)
        for face, (x, y, width, height) in faces:
            face_resized = cv2.resize(face, (160, 160))
            face_embedding = embedder.embeddings([face_resized])[0]
            name = recognize_face(face_embedding)

            cv2.rectangle(frame, (x, y), (x + width, y + height), (0, 255, 0), 2)
            cv2.putText(frame, name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            current_time = time.time()

            if name != "Unknown" and (name not in last_detection_time or current_time - last_detection_time[name] >= 10):
                last_detection_time[name] = current_time  

                person_folder = os.path.join(DETECTED_FACES_FOLDER, name)
                os.makedirs(person_folder, exist_ok=True)

                timestamp = datetime.now().strftime("%d-%m-%Y %H-%M-%S")
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

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()
    cv2.destroyAllWindows()

@app.route('/get_alerts')
def get_alerts():
    global alerts
    return jsonify(alerts)

@app.route('/suspects')
def get_suspects():
    # Find all suspects in the database
    suspects = list(collection.find({}))
    # Convert the ObjectId to string for each document
    for suspect in suspects:
        suspect['_id'] = str(suspect['_id'])
        # Convert binary image data to base64 for display in HTML
        if 'detected_image' in suspect:
            import base64
            suspect['image_b64'] = base64.b64encode(suspect['detected_image']).decode('utf-8')
    
    return render_template('suspects.html', suspects=suspects)

@app.route('/')
def login():
    return render_template('login.html')

@app.route('/landing')
def landing():
    return render_template('landing.html')

@app.route('/live-mon')
def live_mon():
    global alerts
    alerts = []  # Clear previous alerts
    return render_template("livemon.html")

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/contact')
def contact():
    return render_template('contact.html')

@app.route('/new-crim', methods=['GET', 'POST'])
def newcrim():
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

if __name__ == "__main__":
    app.run(debug=True)