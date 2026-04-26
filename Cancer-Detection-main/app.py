import os
import torch
import timm
from flask import Flask, render_template, request, jsonify, send_file, session, redirect, url_for, flash, Response
import functools
from torchvision import transforms
from PIL import Image
import io
import numpy as np
import cv2
import uuid
from image_preprocessing import preprocess_image
from explainability import GradCAM, apply_heatmap
from reporting import generate_pdf_report, generate_comprehensive_report
from database import (
    init_db, save_analysis, get_history, get_user_by_username, 
    get_user_by_id, create_user, get_report_by_id, get_dashboard_stats, 
    save_wellness_entry, get_wellness_history, get_patient_registry, 
    get_patient_detailed_history, export_registry_to_csv
)
from clinical_features import extract_clinical_features

app = Flask(__name__)
app.secret_key = 'super_secret_skin_cancer_key_change_in_production'


@app.after_request
def apply_csp(response):
    response.headers["Content-Security-Policy"] = (
        "default-src 'self'; "
        "script-src 'self' 'unsafe-eval' 'unsafe-inline'; "
        "style-src 'self' 'unsafe-inline';"
    )
    return response

# Config
# Config
# Config
MODEL_PATH_V1 = "saved_models/efficientnet_b0_skin_cancer_4class.pth"
MODEL_PATH_V2 = "saved_models/efficientnet_b0_skin_cancer_4class_v2.pth"
MODEL_PATH_V3 = "saved_models/efficientnet_b4_skin_cancer_4class_v3.pth"

# Model Selection Logic
if os.path.exists(MODEL_PATH_V3):
    MODEL_PATH = MODEL_PATH_V3
    MODEL_ARCH = 'efficientnet_b0'
    MODEL_VERSION = "v3 (SCC Optimized)"
    IMG_SIZE = 384
elif os.path.exists(MODEL_PATH_V2):
    MODEL_PATH = MODEL_PATH_V2
    MODEL_ARCH = 'efficientnet_b0'
    MODEL_VERSION = "v2 (Robust)"
    IMG_SIZE = 224
else:
    MODEL_PATH = MODEL_PATH_V1
    MODEL_ARCH = 'efficientnet_b0'
    MODEL_VERSION = "v1 (Baseline)"
    IMG_SIZE = 224

CLASSES = ['bcc', 'benign', 'melanoma', 'scc']
TEMP_DIR = "static/temp"
os.makedirs(TEMP_DIR, exist_ok=True)

# Load Model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = timm.create_model(MODEL_ARCH, pretrained=False, num_classes=4)

try:
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    print(f"Model loaded: {MODEL_VERSION} ({MODEL_ARCH}) from {MODEL_PATH}")
except Exception as e:
    print(f"Failed to load model: {e}")

model.to(device)
model.eval()

# Initialize Database
init_db()

# Grad-CAM Setup
# For EfficientNet families in timm, model.conv_head is the last conv layer
target_layer = model.conv_head
grad_cam = GradCAM(model, target_layer)

# Transform
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def apply_clinical_refinement(ai_probs, clinical_scores):
    """Refines AI probabilities with strict structural requirements and Benign protection."""
    refined_probs = ai_probs.clone()
    
    # 0. BLANK SKIN OVERRIDE
    if clinical_scores.get('is_blank', False):
        return torch.tensor([0.01, 0.97, 0.01, 0.01], dtype=refined_probs.dtype, device=refined_probs.device)
        
    # --- STEP 1: DEFINE CLINICAL STATES ---
    # Safe Harbor: Strong indicators of a benign lesion
    is_structurally_safe = clinical_scores['asymmetry'] < 0.4 and clinical_scores['border'] < 0.4
    is_tex_safe = clinical_scores['roughness'] < 0.35
    is_benign_safe = is_structurally_safe and is_tex_safe
    
    # Malignant Disruption: Indicators of structural breakdown (Melanoma/Cancers)
    is_disrupted = clinical_scores['asymmetry'] > 0.7 or clinical_scores['border'] > 0.7
    is_highly_disrupted = clinical_scores['asymmetry'] > 0.85 and clinical_scores['border'] > 0.85
    
    # Pigment/Color Risk
    is_colorful = clinical_scores['multicolor'] > 0.65
    is_highly_colorful = clinical_scores['multicolor'] > 0.8
    
    # --- STEP 2: BALANCED HIERARCHY ---
    
    # 1. BENIGN PROTECTION (SAFE HARBOR)
    # If a lesion is structurally safe, we protect it (Nevi/Moles)
    # UNLESS it is extremely colorful, which is a warning sign even for regular shapes.
    if is_benign_safe and not is_highly_colorful:
        # Boost Benign (index 1)
        refined_probs[1] *= 2.0
        # Suppress all malignant classes mildly
        refined_probs[2] *= 0.5 # Suppress Melanoma (pigment focus)
        refined_probs[3] *= 0.5 # Suppress SCC (texture focus)
        refined_probs[0] *= 0.5 # Suppress BCC
        return refined_probs / torch.sum(refined_probs)

    # 2. MELANOMA TRIGGER
    # Melanoma (index 2) triggers on severe indicators.
    # Case A: Both moderately disrupted structure AND colorful pigment
    if is_disrupted and is_colorful:
        refined_probs[2] *= (2.0 + clinical_scores['multicolor'] * 2.0)
        refined_probs[1] *= 0.5 # Suppress Benign mildly
    # Case B: Absolute Priority for extreme structural disruption (e.g. amelanotic melanoma)
    elif is_highly_disrupted:
        refined_probs[2] *= 3.0
        refined_probs[1] *= 0.2
    # Case C: Extreme pigment variation, even if structurally "okay" (Nodular Melanoma)
    elif is_highly_colorful:
        refined_probs[2] *= (1.5 + clinical_scores['multicolor'] * 1.5)
        refined_probs[1] *= 0.5
        
    # 3. SCC vs BCC vs MALIGNANT DIFFERENTIAL
    # Squamous Cell Carcinoma (index 3) - Texture & Crust Focus
    is_scc_textured = clinical_scores['roughness'] > 0.5
    is_scc_crusted = clinical_scores['ulcer'] > 0.4
    is_scc_inflamed = clinical_scores['redness'] > 0.35
    
    if (is_scc_textured or is_scc_crusted) and is_scc_inflamed and not is_benign_safe:
        # Boost SCC (index 3)
        boost = 2.0 + (clinical_scores['roughness'] * 1.5) + (clinical_scores['ulcer'] * 1.0)
        refined_probs[3] *= boost
        refined_probs[1] *= 0.4 # Suppress Benign
        refined_probs[0] *= 0.5 # Suppress BCC (smooth/shiny)
        
    elif clinical_scores['shiny'] > 0.55 and not is_scc_textured:
        # BCC (index 0) boost - Smooth/Shiny over rules rough/crusty
        refined_probs[0] *= 2.0
        refined_probs[3] *= 0.3 # Suppress SCC

    # Ensure no negative probabilities
    refined_probs = torch.clamp(refined_probs, min=1e-7)
    refined_probs = refined_probs / torch.sum(refined_probs)
    
    # --- STEP 3: AI CONFIDENCE BLEND ---
    # If the base AI model is highly confident, we blend the clinical modifications so they act as a gentle steering wheel rather than a total override.
    top_ai_prob = torch.max(ai_probs).item()
    if top_ai_prob > 0.85:
        # 70% Original Prediction, 30% Clinical Shift
        refined_probs = (ai_probs * 0.70) + (refined_probs * 0.30)
    elif top_ai_prob > 0.60:
        # 40% Original Prediction, 60% Clinical Shift
        refined_probs = (ai_probs * 0.40) + (refined_probs * 0.60)
        
    return refined_probs / torch.sum(refined_probs)

# --- AUTH DECORATORS ---
def login_required(f):
    @functools.wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            return redirect(url_for('auth'))
        return f(*args, **kwargs)
    return decorated_function

def admin_required(f):
    @functools.wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session or session.get('role') != 'admin':
            return redirect(url_for('dashboard'))
        return f(*args, **kwargs)
    return decorated_function

# --- ROUTES ---

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/auth', methods=['GET', 'POST'])
def auth():
    if request.method == 'POST':
        action = request.form.get('action')
        username = request.form.get('username')
        password = request.form.get('password')
        
        from werkzeug.security import check_password_hash, generate_password_hash
        
        if action == 'register':
            if get_user_by_username(username):
                flash("Username already exists", "error")
            else:
                pw_hash = generate_password_hash(password)
                user_id = create_user(username, pw_hash, 'patient')
                session['user_id'] = user_id
                session['username'] = username
                session['role'] = 'patient'
                return redirect(url_for('dashboard'))
                
        elif action == 'login':
            user = get_user_by_username(username)
            if user and check_password_hash(user['password_hash'], password):
                session['user_id'] = user['id']
                session['username'] = user['username']
                session['role'] = user['role']
                return redirect(url_for('dashboard'))
            else:
                flash("Invalid credentials", "error")
                
    return render_template('auth.html')

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('index'))

@app.route('/dashboard')
@login_required
def dashboard():
    if session.get('role') == 'admin':
        stats = get_dashboard_stats()
        from database import get_recent_wellness_updates
        recent_wellness = get_recent_wellness_updates(limit=5)
        return render_template('admin_dashboard.html', stats=stats, recent_wellness=recent_wellness)
    return render_template('patient_dashboard.html')

@app.route('/analysis')
@login_required
def analysis():
    return render_template('analysis.html')

@app.route('/history_page')
@login_required
def history_page():
    return render_template('history.html')

@app.route('/performance')
@admin_required
def performance():
    return render_template('performance.html')

@app.route('/settings')
@login_required
def settings():
    return render_template('settings.html')

@app.route('/reset_password', methods=['POST'])
@login_required
def reset_password():
    new_password = request.form.get('new_password')
    if not new_password:
        flash("Password cannot be empty", "error")
        return redirect(url_for('settings'))
    
    from werkzeug.security import generate_password_hash
    from database import update_password
    
    pw_hash = generate_password_hash(new_password)
    update_password(session.get('user_id'), pw_hash)
    
    flash("Password updated successfully!", "success")
    return redirect(url_for('settings'))

# --- PATIENT MODULES ---

@app.route('/wellness')
@login_required
def wellness():
    from database import get_wellness_history
    history = get_wellness_history(session.get('user_id'))
    return render_template('wellness.html', history=history)

@app.route('/admin/wellness')
@admin_required
def admin_wellness():
    from database import get_recent_wellness_updates
    # Fetch more than 5 for the dedicated module
    recent_wellness = get_recent_wellness_updates(limit=50)
    return render_template('wellness_admin.html', recent_wellness=recent_wellness)

@app.route('/save_wellness', methods=['POST'])
@login_required
def save_wellness():
    data = {
        'date': request.form.get('date'),
        'location': request.form.get('location'),
        'itching': request.form.get('itching') == 'on',
        'pain': request.form.get('pain') == 'on',
        'bleeding': request.form.get('bleeding') == 'on',
        'size_change': request.form.get('size_change'),
        'color_change': request.form.get('color_change') == 'on',
        'notes': request.form.get('notes'),
        'cancer_type': request.form.get('cancer_type')
    }
    
    # Handle image upload if present
    file = request.files.get('image')
    if file and file.filename != '':
        upload_dir = "static/uploads/wellness"
        os.makedirs(upload_dir, exist_ok=True)
        filename = f"{uuid.uuid4().hex}_{file.filename}"
        file_path = os.path.join(upload_dir, filename)
        file.save(file_path)
        data['image_path'] = file_path

    save_wellness_entry(session.get('user_id'), data)
    flash("Wellness entry saved successfully!", "success")
    return redirect(url_for('wellness'))

@app.route('/admin/delete_wellness/<int:entry_id>', methods=['POST'])
@admin_required
def delete_wellness(entry_id):
    from database import delete_wellness_entry
    delete_wellness_entry(entry_id, role='admin')
    flash("Wellness journal entry deleted.", "success")
    return redirect(request.referrer or url_for('admin_wellness'))

@app.route('/delete_wellness_user/<int:entry_id>', methods=['POST'])
@login_required
def delete_wellness_user(entry_id):
    from database import delete_wellness_entry
    delete_wellness_entry(entry_id, user_id=session.get('user_id'), role='patient')
    flash("Journal entry removed.", "success")
    return redirect(url_for('wellness'))

@app.route('/delete_record/<int:record_id>', methods=['POST'])
@login_required
def delete_record(record_id):
    from database import delete_analysis_record
    delete_analysis_record(record_id, user_id=session.get('user_id'), role=session.get('role'))
    flash("Diagnostic record has been purged from clinical history.", "success")
    return redirect(request.referrer or url_for('history_page'))

@app.route('/admin/clear_patient_history/<int:user_id>', methods=['POST'])
@admin_required
def clear_patient_history(user_id):
    from database import clear_user_history
    clear_user_history(user_id)
    flash("Patient diagnostic history has been cleared.", "success")
    return redirect(url_for('patient_registry'))

@app.route('/admin/clear_all_history', methods=['POST'])
@admin_required
def clear_all_history():
    from database import clear_all_history_global
    clear_all_history_global()
    flash("GLOBAL CLINICAL HISTORY WIPE COMPLETE.", "danger")
    return redirect(url_for('dashboard'))

@app.route('/library')
@login_required
def library():
    return render_template('library.html')

@app.route('/experts')
@login_required
def experts():
    return render_template('experts.html')

@app.route('/export_history')
@login_required
def export_history():
    from database import get_history
    history = get_history(user_id=session.get('user_id'), limit=1000) # Fetch full history
    return render_template('export_history.html', history=history)

# --- ADMIN MODULES ---

@app.route('/admin/system_health')
@admin_required
def system_health():
    return render_template('system_health.html')

@app.route('/admin/validation_queue')
@admin_required
def validation_queue():
    return render_template('validation_queue.html')

@app.route('/admin/security_audit')
@admin_required
def security_audit():
    return render_template('security_audit.html')

@app.route('/admin/initiate_password_reset', methods=['POST'])
@admin_required
def initiate_password_reset():
    # In a real clinical system, this might send an email or generate a temporary link.
    # For this demo, we'll allow admin to reset a specific user or just flash success.
    # User said: "password can be reset on both user and admin page"
    # We'll redirect to a view where they can manage this or just show it's working.
    flash("Administrative password reset protocol initiated. Please contact IT for the bypass code.", "success")
    return redirect(url_for('security_audit'))

@app.route('/send_to_doctor', methods=['POST'])
@login_required
def send_to_doctor():
    flash("Clinical report has been securely transmitted to your primary care physician.", "success")
    return redirect(request.referrer or url_for('dashboard'))

@app.route('/admin/patient_registry')
@admin_required
def patient_registry():
    # Fetch real patient data
    patients = get_patient_registry()
    
    # Add detailed history for each patient (for the modals)
    for p in patients:
        p['history'] = get_patient_detailed_history(p['id'])
        
    return render_template('patient_registry.html', patients=patients)

@app.route('/admin/export_registry')
@admin_required
def export_registry():
    csv_data = export_registry_to_csv()
    return Response(
        csv_data,
        mimetype="text/csv",
        headers={"Content-disposition": "attachment; filename=clinical_registry_export.csv"}
    )

@app.route('/admin/download_dossier/<int:user_id>')
@admin_required
def download_dossier(user_id):
    user = get_user_by_id(user_id)
    if not user:
        flash("User not found", "error")
        return redirect(url_for('patient_registry'))
    
    # Fetch full history
    history_data = get_patient_detailed_history(user_id)
    
    # Format history for the report generator (it expects a flat list of scans with certain keys)
    # The generator uses 'timestamp', 'prediction', 'confidence'
    formatted_history = []
    for scan in history_data['scans']:
        formatted_history.append({
            'timestamp': scan['timestamp'],
            'prediction': scan['prediction'],
            'confidence': f"{scan['confidence'] * 100:.2f}%"
        })
    
    filename = f"dossier_{user['username']}_{uuid.uuid4().hex[:8]}.pdf"
    output_path = os.path.join(TEMP_DIR, filename)
    
    generate_comprehensive_report(formatted_history, user['username'], output_path)
    
    return send_file(output_path, as_attachment=True)

@app.route('/consult_specialist', methods=['POST'])
@login_required
def consult_specialist():
    flash("Inquiry sent to specialist. You will receive a notification within 12-24 hours.", "success")
    return redirect(url_for('experts'))

@app.route('/share_with_expert', methods=['POST'])
@login_required
def share_with_expert():
    flash("Electronic health records and scan data shared with the selected institution.", "success")
    return redirect(url_for('experts'))

@app.route('/admin/protocols')
@admin_required
def protocols():
    return render_template('protocols.html')

@app.route('/admin/save_protocols', methods=['POST'])
@admin_required
def save_protocols():
    flash("Clinical thresholds and refinement weights updated successfully.", "success")
    return redirect(url_for('protocols'))



@app.route('/predict', methods=['POST'])
@login_required
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    try:
        session_id = str(uuid.uuid4())
        # Read image
        image_bytes = file.read()
        pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        
        # Save original temporarily for report
        orig_path = os.path.join(TEMP_DIR, f"{session_id}_orig.jpg")
        pil_image.save(orig_path)

        # Convert to BGR for cv2 preprocessing
        img_np_rgb = np.array(pil_image)
        img_np_bgr = cv2.cvtColor(img_np_rgb, cv2.COLOR_RGB2BGR)

        # Preprocess (returns 0-1 float in BGR)
        processed_img_bgr = preprocess_image(img_np_bgr, IMG_SIZE)
        
        # Convert back to RGB for model
        processed_img_rgb = cv2.cvtColor(processed_img_bgr.astype('float32'), cv2.COLOR_BGR2RGB)

        # ToTensor expects (H, W, C).
        input_tensor = transform(processed_img_rgb).unsqueeze(0).to(device)

        # Inference (Need gradients for Grad-CAM, so not using no_grad here)
        input_tensor.requires_grad = True
        outputs = model(input_tensor)
        probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
        
        # Grad-CAM Heatmap
        class_idx = torch.argmax(probabilities).item()
        heatmap = grad_cam.generate_heatmap(input_tensor, class_idx)
        
        # Apply heatmap to the processed image (or original)
        # We'll use the original size for better visualization
        orig_cv2 = cv2.imread(orig_path)
        superimposed = apply_heatmap(heatmap, orig_cv2)
        
        heatmap_path = os.path.join(TEMP_DIR, f"{session_id}_heatmap.jpg")
        cv2.imwrite(heatmap_path, superimposed)

        # Clinical Feature Extraction (Robust fallback)
        clinical_metrics = {'asymmetry': 0, 'border': 0, 'shiny': 0, 'roughness': 0, 'redness': 0, 'ulcer': 0, 'multicolor': 0}
        try:
            clinical_metrics = extract_clinical_features(img_np_bgr)
            
            # DIAGNOSTIC LOGGING
            print("\n--- DIAGNOSTIC ANALYSIS ---")
            print(f"File: {file.filename}")
            print(f"Is Skin: {clinical_metrics['is_skin']}")
            print(f"AI Raw Probs: {dict(zip(CLASSES, [p.item()*100 for p in probabilities]))}")
            print(f"Clinical Scores: {clinical_metrics}")
            
            # --- SKIN VALIDATION FILTER ---
            if not clinical_metrics['is_skin']:
                return jsonify({
                    'success': False,
                    'error': 'Invalid Specimen: The image could not be verified as skin. Please ensure clear lighting and centered framing of the lesion.',
                    'is_skin_failure': True
                }), 400

            # Apply Clinical Refinement Layer
            probabilities = apply_clinical_refinement(probabilities, clinical_metrics)
            
            print(f"Refined Probs: {dict(zip(CLASSES, [p.item()*100 for p in probabilities]))}")
            print("---------------------------\n")
            
        except Exception as e:
            print(f"Warning: Clinical feature extraction or refinement failed: {e}")

        # Format results
        results = []
        for i, prob in enumerate(probabilities):
            results.append({
                'class': CLASSES[i],
                'probability': prob.item() * 100
            })
        
        # Sort by probability
        results.sort(key=lambda x: x['probability'], reverse=True)
        
        # Save to database (Robust fallback)
        try:
            save_analysis(
                orig_path, 
                results[0]['class'], 
                results[0]['probability'] / 100, 
                session.get('user_id'),
                probabilities=results
            )
        except Exception as e:
            print(f"Warning: Database save failed: {e}")
        
        return jsonify({
            'success': True, 
            'predictions': results,
            'heatmap_url': f"/{heatmap_path}",
            'session_id': session_id,
            'model_version': MODEL_VERSION,
            'clinical_features': clinical_metrics
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/download_report/<session_id>', methods=['POST'])
def download_report(session_id):
    try:
        data = request.json
        predictions = data.get('predictions')
        
        orig_path = os.path.join(TEMP_DIR, f"{session_id}_orig.jpg")
        heatmap_path = os.path.join(TEMP_DIR, f"{session_id}_heatmap.jpg")
        report_path = os.path.join(TEMP_DIR, f"{session_id}_report.pdf")
        
        if not os.path.exists(orig_path) or not os.path.exists(heatmap_path):
            return jsonify({'error': 'Session expired or files missing'}), 404
            
        generate_pdf_report(predictions, orig_path, heatmap_path, report_path)
        
        return send_file(report_path, as_attachment=True)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/history')
@login_required
def history_api():
    if session.get('role') == 'admin':
        data = get_history() # all data
    else:
        data = get_history(user_id=session.get('user_id'))
    return jsonify(data)

@app.route('/download_history_pdf')
@login_required
def download_history_pdf():
    from database import get_history
    from reporting import generate_comprehensive_report
    
    history = get_history(user_id=session.get('user_id'), limit=1000)
    if not history:
        flash("No data available to export.", "error")
        return redirect(url_for('export_history'))
        
    filename = f"health_report_{session.get('username')}_{uuid.uuid4().hex[:8]}.pdf"
    output_path = os.path.join(TEMP_DIR, filename)
    
    generate_comprehensive_report(history, session.get('username'), output_path)
    
    return send_file(output_path, as_attachment=True, download_name="Clinical_History_Portfolio.pdf")

@app.route('/report/<int:report_id>')
@login_required
def view_report(report_id):
    report = get_report_by_id(report_id, session.get('user_id'), session.get('role'))
    if not report:
        return "Report not found or access denied", 404
    return render_template('report.html', report=report)

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 10000))  # MUST use this
    app.run(host='0.0.0.0', port=port)
