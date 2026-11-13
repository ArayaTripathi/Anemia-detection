import os
import uuid
from flask import Flask, request, render_template, jsonify, send_file, send_from_directory, url_for
from werkzeug.utils import secure_filename
from PIL import Image

from model_service import predict_anemia_sample, CLASSES, ANEMIA_INFO
from pdf import generate_pdf_report

app = Flask(__name__, static_folder='static')

UPLOAD_FOLDER = 'uploads'
REPORTS_FOLDER = 'reports'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'jfif', 'tiff', 'tif', 'bmp', 'webp'}

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(REPORTS_FOLDER, exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/home')
def home():
    return render_template('home.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/uploads/<path:filename>')
def uploaded_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Form inputs
        name = request.form.get('name', 'Patient').strip() or 'Patient'
        age = request.form.get('age', '')
        gender = request.form.get('gender', '')
        blood_type = request.form.get('blood_type', '')
        hemoglobin = request.form.get('hemoglobin', '')
        mcv = request.form.get('mcv', '')
        rbc_count = request.form.get('rbc_count', '')
        mch = request.form.get('mch', '')

        # File validation
        if 'image' not in request.files:
            warning = "No image file provided in request."
            return render_template('index.html', warning=warning)

        file = request.files['image']
        if file.filename == '' or not allowed_file(file.filename):
            warning = "Please upload a valid blood smear image file (PNG, JPG, TIFF, etc.)."
            return render_template('index.html', warning=warning)

        # Unique safe filename
        ext = file.filename.rsplit('.', 1)[1].lower() if '.' in file.filename else 'jpg'
        unique_filename = f"{secure_filename(os.path.splitext(file.filename)[0])}_{uuid.uuid4().hex[:8]}.{ext}"
        filepath = os.path.join(UPLOAD_FOLDER, unique_filename)
        file.save(filepath)

        # Verify image opens with PIL
        try:
            with Image.open(filepath) as test_img:
                test_img.verify()
        except Exception:
            if os.path.exists(filepath):
                os.remove(filepath)
            warning = "Corrupted or unreadable image file. Please provide a valid microscope smear image."
            return render_template('index.html', warning=warning)

        # Run AI prediction using fine‑tuned model – protect against image loading errors
        try:
            prediction_result = predict_anemia_sample(
                img_path_or_pil=filepath,
                hemoglobin=hemoglobin,
                mcv=mcv,
                rbc_count=rbc_count,
                mch=mch,
                h5_path='M_Anemia_finetuned.h5'
            )
        except Exception as e:
            # Remove the uploaded file to avoid clutter
            if os.path.exists(filepath):
                os.remove(filepath)
            warning = f"Processing error: {e}. Ensure the uploaded file is a valid image."
            return render_template('index.html', warning=warning)

        predicted_class = prediction_result['predicted_class']
        confidence_val = prediction_result['confidence']
        confidence_str = prediction_result['confidence_str']
        all_probs = prediction_result['all_probabilities']
        morphology = prediction_result['morphology']
        recommendations = prediction_result['recommendations']
        cbc_notes = prediction_result['cbc_corroboration']

        # Generate PDF lab report
        pdf_path = generate_pdf_report(
            patient_name=name,
            age=age,
            gender=gender,
            predicted_class=predicted_class,
            class_prob=confidence_val,
            rbc_count=rbc_count,
            blood_type=blood_type,
            hemoglobin=hemoglobin,
            mch=mch,
            mcv=mcv,
            recommendations=recommendations,
            morphology=morphology
        )
        pdf_filename = os.path.basename(pdf_path)
        image_url = url_for('uploaded_file', filename=unique_filename)

        # Check if request was JSON/Fetch
        if request.headers.get('X-Requested-With') == 'XMLHttpRequest' or request.is_json:
            return jsonify({
                'success': True,
                'anemiaType': predicted_class,
                'confidence': confidence_str,
                'probability': confidence_val,
                'allProbabilities': all_probs,
                'morphology': morphology,
                'recommendations': recommendations,
                'imageUrl': image_url,
                'pdfFilename': pdf_filename,
                'cbcNotes': cbc_notes
            })

        # Render complete website with results
        return render_template(
            'index.html',
            result=predicted_class,
            probability=confidence_str,
            all_probabilities=all_probs,
            pdf_filename=pdf_filename,
            uploaded_image_url=image_url,
            morphology=morphology,
            recommendations=recommendations,
            cbc_corroboration=cbc_notes,
            patient_name=name,
            age=age,
            gender=gender,
            blood_type=blood_type,
            hemoglobin=hemoglobin,
            mcv=mcv,
            rbc_count=rbc_count,
            mch=mch
        )

    except Exception as e:
        app.logger.error(f"Prediction error: {e}", exc_info=True)
        return render_template('index.html', warning=f"Processing error: {str(e)}")


@app.route('/download_report/<filename>')
def download_report(filename):
    safe_name = secure_filename(filename)
    pdf_path = os.path.join(REPORTS_FOLDER, safe_name)
    if not os.path.exists(pdf_path):
        return "Report file not found.", 404
    return send_file(pdf_path, as_attachment=True)


if __name__ == '__main__':
    print("Starting Anemia Detection Flask server on port 5000...")
    app.run(host='0.0.0.0', port=5000, debug=False)
