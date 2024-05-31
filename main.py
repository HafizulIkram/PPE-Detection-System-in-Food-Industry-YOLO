# webcam.py
from flask import Flask, render_template, Response, jsonify
import argparse
from image_detection import generate_frames, release_webcam, get_detected_classes, initialize_webcam
from graph import plot_png

app = Flask(__name__)
app.secret_key = 'your_secret_key'

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

@app.route("/video_feed")
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/stop_webcam')
def stop_webcam():
    release_webcam()
    return ('', 204)

@app.route('/detected_classes')
def detected_classes():
    classes = get_detected_classes()
    return jsonify(classes)

@app.route('/plot')
def plot():
    return plot_png()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Flask app exposing yolov7 models")
    parser.add_argument("--port", default=5000, type=int, help="port number")
    args = parser.parse_args()
    app.run(host="0.0.0.0", port=args.port)
