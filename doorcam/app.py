from flask import Flask, request, render_template, jsonify
import pandas as pd
from predict import Identifier, extract_name_from_path
from utils import get_config
from PIL import Image
from io import BytesIO
from werkzeug.utils import secure_filename
import os
from glob import glob
from datetime import datetime, timedelta

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

config = get_config()
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = "static/people/"

predictor = Identifier(config)


@app.route('/')
def home():
    directory = "people"
    entry_record = predictor.load_record()
    image_files = os.listdir(os.path.join("static", directory))
    images = [os.path.join(directory, file) for file in image_files if file.split('.')[-1].lower() in ALLOWED_EXTENSIONS]
    return render_template('display.html',  table=entry_record.to_html(classes='table table-striped', columns=["Name", "Timestamp"], index=False), images=images)

@app.route('/update', methods=['POST'])
def update_data():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    if file:
        # Read the image via PIL
        image = Image.open(BytesIO(file.read()))
        all_records = predictor.load_record()
        record = predictor.inference(image)
        print(" ")
        print(" ")
        print(" ")
        print(" ")
        print(" ")
        print(record)
        print(" ")
        print(" ")
        print(" ")
        print(" ")
        print(" ")
        print(" ")
        
        if record is not None:
            filtered_records = all_records[all_records["Name"]==record["Name"][0]]
            last_timestamp = filtered_records['Timestamp'].max()
            if last_timestamp + timedelta(minutes=10) <= record['Timestamp'][0] or filtered_records.empty:
                predictor.add_entryrecord(record)
                entry_record = predictor.load_record()
        return jsonify({'message': 'Image received successfully'}), 200


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        file = request.files['file']
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            return 'File uploaded successfully'
    return render_template('upload.html')


@app.route('/delete/<filename>', methods=['GET', 'POST'])
def delete_file(filename):
    if '..' in filename or filename.startswith('/'):
        return "Invalid filename", 400
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)

    if os.path.exists(file_path):
        os.remove(file_path)
        return "File deleted successfully", 200
    else:
        return "File not found", 404

if __name__ == '__main__':
    app.run(debug=True)