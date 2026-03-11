# app.py
from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename
import os
import tempfile
from ml_core import predict_brain_tumor_web

# --- Flask Setup ---
app = Flask(__name__, template_folder='.')
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file part in the request"}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    if file and allowed_file(file.filename):
        temp_dir = tempfile.mkdtemp()
        filename = secure_filename(file.filename)
        filepath = os.path.join(temp_dir, filename)

        try:
            file.save(filepath)

            # ML CORE CALL
            results = predict_brain_tumor_web(filepath)

            # Cleanup
            os.remove(filepath)
            os.rmdir(temp_dir)

            return jsonify(results)

        except Exception as e:
            import traceback
            traceback.print_exc()

            if os.path.exists(filepath):
                os.remove(filepath)
            if os.path.exists(temp_dir):
                os.rmdir(temp_dir)

            return jsonify({"error": str(e)}), 500

    return jsonify({"error": "Invalid file type. Upload PNG/JPG only."}), 400


if __name__ == '__main__':
    from ml_core import initialize_models
    if initialize_models():
        print("\n🚀 Starting Flask app. Navigate to http://127.0.0.1:5000\n")

        # 🔥 MOST IMPORTANT FIX
        app.run(debug=False, use_reloader=False)

    else:
        print("\n🚨 ERROR: Models failed to load.\n")
