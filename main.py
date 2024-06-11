from flask import Flask, render_template, Response, jsonify, request, send_file, redirect, send_from_directory, abort, flash, url_for
import argparse
from yoloWebcam import detect_cam
from report import generate_report, create_pdf_report
from model import db, CompletionStatus, Class, DetectionEvent
from report import generate_report, create_pdf_report, create_bar_chart, create_pie_chart
from imageDetection import detect_objects
import os
from werkzeug.utils import secure_filename
import cv2
import tempfile
from datetime import datetime

app = Flask(__name__)
app.secret_key = 'your_secret_key'

app.config['UPLOAD_FOLDER'] = 'C:\\Users\\hafiz\\Desktop\\newInterface\\uploads'
app.config['PREDICTIONS_FOLDER'] = 'C:\\Users\\hafiz\\Desktop\\newInterface\\predictions'
app.config['ALLOWED_EXTENSIONS'] = {'jpg', 'jpeg', 'png'}

# Configure the SQLAlchemy part of the app instance
app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql+pymysql://root@localhost/yolodetection'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Initialize the SQLAlchemy db instance with the app
db.init_app(app)

with app.app_context():
    db.create_all()


if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

if not os.path.exists(app.config['PREDICTIONS_FOLDER']):
    os.makedirs(app.config['PREDICTIONS_FOLDER'])

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']


is_camera_active = False
camera = None
detected_classes = []
completion_status = []


# Global variables to store chart paths
chart_dir = tempfile.gettempdir()
class_chart_filename = None
status_chart_filename = None

def initialize_camera():
    global camera, is_camera_active
    if camera is None or not camera.isOpened():
        camera = cv2.VideoCapture(0)
    is_camera_active = True

def release_camera():
    global camera, is_camera_active
    if camera is not None and camera.isOpened():
        camera.release()
    is_camera_active = False

@app.route("/")
def home():
    return render_template('dashboard.html')

@app.route("/index")
def index():
    return render_template('index.html')


@app.route("/webcam")
def webcam():
    return render_template('webcam.html')

@app.route("/webDete")
def webDete():
    return render_template('webpath.html')

@app.route("/webDete_feed")
def webDete_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route("/graphDis")
def graphDis():
    return render_template('graphDisplay.html')

def gen_frames(confidence_threshold=0.25):
    global detected_classes, completion_status, is_camera_active
    initialize_camera()
    while is_camera_active:
        success, frame = camera.read()
        if not success:
            break
        else:
            # Detect objects in the frame
            frame, detected_classes, completion_status = detect_cam(frame, confidence_threshold)

            # Store the results in the database within the application context
            with app.app_context():
                store_detection_results(detected_classes, completion_status)

            # Encode the frame in JPEG format
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()

            # Yield the frame in byte format
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    # Clear detected classes and completion status when the loop exits
    detected_classes = []
    completion_status = []

def store_detection_results(classes, status):
    # Retrieve completion status
    status_record = CompletionStatus.query.filter_by(status=status).first()
    if not status_record:
        status_record = CompletionStatus(status=status)
        db.session.add(status_record)
        db.session.commit()

    for class_name in classes:
        # Retrieve detected class
        class_record = Class.query.filter_by(name=class_name).first()
        if not class_record:
            raise ValueError(f"Class '{class_name}' not found in the database.")

        # Store detection event
        detection_event = DetectionEvent(
            completion_status_id=status_record.id,
            classID=class_record.id  # Ensure the correct foreign key reference
        )
        db.session.add(detection_event)
        db.session.commit()



@app.route('/video_feed')
def video_feed():
    confidence_threshold = float(request.args.get('confidence', 0.25))
    return Response(gen_frames(confidence_threshold),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/detected_classes')
def get_detected_classes():
    global detected_classes, completion_status
    return jsonify({
        'detected_classes': detected_classes,
        'completion_status': completion_status,
    })

@app.route('/start_camera')
def start_camera():
    initialize_camera()
    return jsonify({'status': 'Camera started'})

@app.route('/stop_camera')
def stop_camera():
    release_camera()
    return jsonify({'status': 'Camera stopped'})


@app.route("/report")
def report():
    return render_template('reportDisplay.html')


@app.route('/generate_report', methods=['POST'])
def generate_report_view():
    start_date_str = request.form['start_date']
    end_date_str = request.form['end_date']
    start_date = datetime.strptime(start_date_str, '%Y-%m-%d')
    end_date = datetime.strptime(end_date_str, '%Y-%m-%d')
    
    if start_date > datetime.now():
        flash("Start date cannot be later than today.")
        return redirect(url_for('report_display'))

    class_counts, status_counts, analysis_paragraph = generate_report(start_date, end_date)
    
    if not class_counts and not status_counts:
        flash("No data available for the chosen date range. Please enter a different date range.")
        return redirect(url_for('report_display'))
    
    return render_template('reportDisplay.html', 
                           start_date=start_date, 
                           end_date=end_date, 
                           analysis_paragraph=analysis_paragraph)

@app.route('/class_chart')
def class_chart():
    chart_path = os.path.join(tempfile.gettempdir(), 'Class_Distribution.png')
    return send_file(chart_path, mimetype='image/png')

@app.route('/status_chart')
def status_chart():
    chart_path = os.path.join(tempfile.gettempdir(), 'Completion_Status_Distribution.png')
    return send_file(chart_path, mimetype='image/png')

@app.route('/download_report', methods=['POST'])
def download_report():
    start_date_str = request.form['start_date']
    end_date_str = request.form['end_date']
    start_date = datetime.strptime(start_date_str, '%Y-%m-%d')
    end_date = datetime.strptime(end_date_str, '%Y-%m-%d')

    class_counts, status_counts, analysis_paragraph = generate_report(start_date, end_date)

    pdf_path = os.path.join(tempfile.gettempdir(), 'report.pdf')
    create_pdf_report(class_counts, status_counts, analysis_paragraph, start_date, end_date, pdf_path)

    return send_file(pdf_path, as_attachment=True)


@app.route('/index', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            try:
                predicted_image_path, class_labels, completion_status = detect_objects(filepath, app.config['PREDICTIONS_FOLDER'])

                # Save the status and detected classes to the database
                completion_status_entry = CompletionStatus.query.filter_by(status=completion_status).first()
                if not completion_status_entry:
                    completion_status_entry = CompletionStatus(status=completion_status)
                    db.session.add(completion_status_entry)
                    db.session.commit()

                for label in class_labels:
                    class_entry = Class.query.filter_by(name=label).first()
                    if not class_entry:
                        class_entry = Class(name=label)
                        db.session.add(class_entry)
                        db.session.commit()

                    detection_event = DetectionEvent(
                        completion_status_id=completion_status_entry.id,
                        classID=class_entry.id
                    )
                    db.session.add(detection_event)

                db.session.commit()

                return render_template('index.html', uploaded_image_path=filename, predicted_image_path=os.path.basename(predicted_image_path), class_labels=class_labels, completion_status=completion_status)
            except Exception as e:
                print(f"Error during detection: {e}")
                return redirect(request.url)
    return render_template('index.html')

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/predictions/<path:filename>')
def prediction_file(filename):
    try:
        return send_from_directory(app.config['PREDICTIONS_FOLDER'], filename)
    except FileNotFoundError:
        abort(404)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Flask app exposing yolov7 models")
    parser.add_argument("--port", default=5000, type=int, help="port number")
    args = parser.parse_args()
    app.run(host="0.0.0.0", port=args.port)
