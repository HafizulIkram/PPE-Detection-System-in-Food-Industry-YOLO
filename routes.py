from flask import render_template, request, Response, jsonify, send_from_directory
import os
from webcam import generate_frames, release_webcam, get_detected_classes
from graph import plot_png
from image_detect import handle_prediction

def init_routes(app):
    @app.route("/")
    def home():
        return render_template('dashboard.html')

    @app.route("/index", methods=["GET", "POST"])
    def predict_img():
        class_labels = []  # Initialize an empty list to store predicted classes
        if request.method == "POST":
            if 'file' in request.files:
                f = request.files['file']
                basepath = os.path.dirname(__file__)
                filepath = os.path.join(basepath, 'uploads', f.filename)
                print("upload folder is ", filepath)
                f.save(filepath)

            predict_img.imgpath = f.filename


            file_extension = f.filename.rsplit('.', 1)[1].lower()    
            if file_extension == 'jpg':
                process = Popen(["python", "detect.py", '--source', filepath, "--weights","YoloWeight\yolov7W6.pt", "--save-txt"], shell=True)
                process.wait()
            
            
        # Render the template with no image paths if no file is uploaded or if it's a GET request
        return render_template('index.html', uploaded_image_path=None, predicted_image_path=None)

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

    @app.route('/<path:filename>')
    def display(filename):
        folder_path = 'runs\detect'
        subfolders = [f for f in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, f))]    
        latest_subfolder = max(subfolders, key=lambda x: os.path.getctime(os.path.join(folder_path, x)))    
        directory = os.path.join(folder_path, latest_subfolder)
        print("printing directory: ", directory) 

        # Check if the requested file is an uploaded image
        if filename.startswith('uploads'):
            # Serve the uploaded image
            return send_from_directory(os.path.dirname(__file__), filename, as_attachment=True, environ=request.environ)
    
        # Check if the requested file is a predicted image
        elif filename.startswith('runs\detect') and filename.endswith(('.jpg', '.png', '.jpeg')):
            # Serve the predicted image
            filename = filename.replace('\\', '/')  # Replace backslashes with forward slashes
            predicted_image_path = os.path.join(directory, os.path.basename(filename))
            if os.path.exists(predicted_image_path):
                return send_from_directory(directory, os.path.basename(filename), as_attachment=True, environ=request.environ)
            else:
                return "Predicted image not found"
    
        else:
            return "Invalid file path or file type"