# app.py
import os
import io
import base64
import time
import requests
import numpy as np
import tensorflow as tf
from PIL import Image, ImageDraw, ImageFont
from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_cors import CORS
from pathlib import Path as _Path
from sklearn.cluster import KMeans
from datetime import datetime
import cv2
import textwrap
import re
import logging

# Matplotlib safe backend (used for small visualizations)
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

# ---------------- CONFIG ----------------
KMEANS_HOST = "http://localhost:5001"   # optional external kmeans forward target
FORWARD_DIR = _Path("static/forwarded")
OUTPUT_DIR = _Path("static/outputs")
UPLOAD_DIR = _Path("static/uploads")    # NEW: store original uploads here

FORWARD_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
FONT_DIR = _Path("./fonts")
FONT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------- Logging ----------------
logging.basicConfig(level=logging.INFO)

# ---------------- 1. Load Binary SAR Classifier ----------------
SAR_MODEL_PATH = "sar_binary_classifier.h5"
if not os.path.exists(SAR_MODEL_PATH):
    raise FileNotFoundError(f"SAR classifier not found at {SAR_MODEL_PATH}")
sar_binary_model = tf.keras.models.load_model(SAR_MODEL_PATH)
IMG_SIZE_BINARY = (128, 128)

# ---------------- 2. Load Colorization Generator ----------------
IMG_HEIGHT = 256
IMG_WIDTH = 256
OUTPUT_CHANNELS = 3
CHECKPOINT_DIR = r"D:\final year project\training_checkpoints"

def downsample(filters, size, apply_batchnorm=True):
    initializer = tf.random_normal_initializer(0., 0.02)
    result = tf.keras.Sequential([
        tf.keras.layers.Conv2D(filters, size, strides=2, padding='same',
                               kernel_initializer=initializer, use_bias=False)
    ])
    if apply_batchnorm:
        result.add(tf.keras.layers.BatchNormalization())
    result.add(tf.keras.layers.LeakyReLU())
    return result

def upsample(filters, size, apply_dropout=False):
    initializer = tf.random_normal_initializer(0., 0.02)
    result = tf.keras.Sequential([
        tf.keras.layers.Conv2DTranspose(filters, size, strides=2, padding='same',
                                        kernel_initializer=initializer, use_bias=False),
        tf.keras.layers.BatchNormalization()
    ])
    if apply_dropout:
        result.add(tf.keras.layers.Dropout(0.5))
    result.add(tf.keras.layers.ReLU())
    return result

def Generator():
    inputs = tf.keras.layers.Input(shape=[IMG_HEIGHT, IMG_WIDTH, 3])
    down_stack = [
        downsample(64, 4, apply_batchnorm=False),
        downsample(128, 4),
        downsample(256, 4),
        downsample(512, 4),
        downsample(512, 4),
        downsample(512, 4),
        downsample(512, 4),
        downsample(512, 4),
    ]
    up_stack = [
        upsample(512, 4, apply_dropout=True),
        upsample(512, 4, apply_dropout=True),
        upsample(512, 4, apply_dropout=True),
        upsample(512, 4),
        upsample(256, 4),
        upsample(128, 4),
        upsample(64, 4),
    ]
    initializer = tf.random_normal_initializer(0., 0.02)
    last = tf.keras.layers.Conv2DTranspose(OUTPUT_CHANNELS, 4, strides=2, padding='same',
                                          kernel_initializer=initializer, activation='tanh')
    x = inputs
    skips = []
    for down in down_stack:
        x = down(x)
        skips.append(x)
    skips = reversed(skips[:-1])
    for up, skip in zip(up_stack, skips):
        x = up(x)
        x = tf.keras.layers.Concatenate()([x, skip])
    x = last(x)
    return tf.keras.Model(inputs=inputs, outputs=x)

# build generator and restore checkpoint
generator = Generator()
checkpoint = tf.train.Checkpoint(generator=generator)
latest_ckpt = tf.train.latest_checkpoint(CHECKPOINT_DIR)
if latest_ckpt:
    checkpoint.restore(latest_ckpt).expect_partial()
    logging.info("Loaded checkpoint: %s", latest_ckpt)
else:
    logging.warning("Warning: No checkpoint found at %s", CHECKPOINT_DIR)

# ---------------- 3. Flask App ----------------
app = Flask(__name__, template_folder="templates", static_folder="static")
CORS(app)
app.config['TRAP_HTTP_EXCEPTIONS'] = True

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/results')
def results_page():
    status = request.args.get('status')
    message = request.args.get('message')
    return render_template('results.html', status=status, message=message)

@app.route('/results.html')
def results_html_page():
    status = request.args.get('status')
    message = request.args.get('message')
    return render_template('results.html', status=status, message=message)

@app.route('/favicon.ico')
def favicon():
    fav = os.path.join(app.static_folder or 'static', 'favicon.ico')
    if os.path.exists(fav):
        return send_from_directory(app.static_folder, 'favicon.ico')
    return ('', 204)

# ---------------- Helpers ----------------
def is_sar_image_bytes(image_bytes: bytes):
    img = tf.image.decode_image(image_bytes, channels=3, expand_animations=False)
    img.set_shape([None, None, 3])
    img = tf.image.resize(img, IMG_SIZE_BINARY)
    img = img / 255.0
    img = tf.expand_dims(img, axis=0)
    prediction = sar_binary_model.predict(img)[0][0]
    return prediction >= 0.5, float(prediction)

def is_sar_image(image_path: str):
    img = tf.io.read_file(image_path)
    img = tf.image.decode_image(img, channels=3, expand_animations=False)
    img.set_shape([None, None, 3])
    img = tf.image.resize(img, IMG_SIZE_BINARY)
    img = img / 255.0
    img = tf.expand_dims(img, axis=0)
    prediction = sar_binary_model.predict(img)[0][0]
    return prediction >= 0.5, float(prediction)

def extract_image_from_request():
    data_json = request.get_json(silent=True)
    image_b64 = None
    if data_json and isinstance(data_json, dict) and 'image' in data_json:
        raw = data_json['image']
        image_b64 = raw.split(',', 1)[1] if ',' in raw else raw
    else:
        if 'image' in request.form:
            raw = request.form.get('image')
            image_b64 = raw.split(',', 1)[1] if ',' in raw else raw
        elif 'image' in request.files or 'file' in request.files:
            upload = request.files.get('image') or request.files.get('file')
            file_bytes = upload.read()
            image_b64 = base64.b64encode(file_bytes).decode()
    if not image_b64:
        raise ValueError("No image found in JSON/form/files.")
    try:
        image_bytes = base64.b64decode(image_b64)
    except Exception as e:
        raise ValueError(f"Base64 decode failed: {e}")
    pil_image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    return pil_image, image_bytes

# ---------------- 4. Analyze (SAR) ----------------
@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        pil_image, raw_bytes = extract_image_from_request()
    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    try:
        is_sar, conf = is_sar_image_bytes(raw_bytes)
    except Exception as e:
        app.logger.exception("SAR check failed")
        return jsonify({'error': 'SAR check failed', 'detail': str(e)}), 500
    return jsonify({'is_sar': bool(is_sar), 'sar_conf': float(conf)}), 200

# ---------------- 5. Colorize (Saves original to uploads/) ----------------
@app.route('/colorize', methods=['POST'])
def colorize_and_forward():
    try:
        pil_image, raw_bytes = extract_image_from_request()
    except ValueError as e:
        return jsonify({'error': str(e)}), 400

    # Use a single timestamp so orig and colorized filenames match
    ts = int(time.time())

    # Save original permanently in static/uploads/
    orig_filename = f"original_{ts}.png"
    orig_path = UPLOAD_DIR / orig_filename
    try:
        pil_image.save(orig_path)
    except Exception as e:
        app.logger.exception("Failed to save original upload: %s", e)

    # Also keep a temp file for SAR checking if needed
    temp_image_path = f"temp_colorize_image_{ts}.png"
    pil_image.save(temp_image_path)

    try:
        sar_check, sar_conf = is_sar_image(temp_image_path)
    except Exception as e:
        app.logger.exception("SAR check failed")
        try: os.remove(temp_image_path)
        except Exception: pass
        return jsonify({'error': 'SAR check failed', 'detail': str(e)}), 500

    if not sar_check or sar_conf < 0.7:
        try: os.remove(temp_image_path)
        except Exception: pass
        return jsonify({'error': "Image is not a valid SAR image.", 'sar_conf': sar_conf}), 400

    try:
        pil_resized = pil_image.resize((IMG_WIDTH, IMG_HEIGHT))
        input_image = np.array(pil_resized).astype(np.float32)
        input_image_normalized = (input_image / 127.5) - 1.0
        input_batch = np.expand_dims(input_image_normalized, 0)
        prediction = generator(input_batch, training=True)
        output_01 = (prediction[0] * 0.5 + 0.5)
        output_uint8 = tf.cast(tf.round(output_01 * 255.0), tf.uint8).numpy()
        output_pil = Image.fromarray(output_uint8)
        buffered = io.BytesIO()
        output_pil.save(buffered, format="PNG")
        output_b64 = "data:image/png;base64," + base64.b64encode(buffered.getvalue()).decode()
    except Exception as e:
        app.logger.exception("Colorization failed")
        try: os.remove(temp_image_path)
        except Exception: pass
        return jsonify({'error': 'Colorization failed', 'detail': str(e)}), 500

    forwarded_name = f"colorized_{ts}.png"
    forwarded_path = FORWARD_DIR / forwarded_name
    with open(forwarded_path, "wb") as f:
        f.write(buffered.getvalue())

    forward_result = None
    try:
        upload_url = KMEANS_HOST.rstrip("/") + "/upload"
        files = {"image": (forwarded_path.name, forwarded_path.read_bytes(), "image/png")}
        resp = requests.post(upload_url, files=files, timeout=20)
        try:
            j = resp.json()
        except Exception:
            j = {'status': resp.status_code, 'text': resp.text}
        forward_result = {'ok': 200 <= resp.status_code < 300, 'status_code': resp.status_code, 'response': j}
    except Exception as e:
        forward_result = {'ok': False, 'error': str(e)}

    try:
        if os.path.exists(temp_image_path):
            os.remove(temp_image_path)
    except Exception:
        pass

    public_url = f"/forwarded/{forwarded_name}"
    # Return orig_filename as well so frontend can forward/redirect with it if desired
    return jsonify({'colorized_image': output_b64, 'saved_path': str(forwarded_path),
                    'filename': forwarded_name, 'public_url': public_url,
                    'orig_filename': orig_filename,
                    'forwarded_to_kmeans': forward_result}), 200

# ---------------- 6. Upload ----------------
@app.route('/upload', methods=['POST'])
def upload():
    try:
        if 'image' in request.files or 'file' in request.files:
            upload_file = request.files.get('image') or request.files.get('file')
            file_bytes = upload_file.read()
            incoming_name = os.path.basename(upload_file.filename or f"upload_{int(time.time())}.png")
            filename = incoming_name
            forwarded_path = FORWARD_DIR / filename
            if forwarded_path.exists():
                filename = f"{int(time.time())}_{filename}"
                forwarded_path = FORWARD_DIR / filename
            with open(forwarded_path, "wb") as f:
                f.write(file_bytes)
        else:
            data_json = request.get_json(silent=True)
            if data_json and isinstance(data_json, dict) and 'image' in data_json:
                raw = data_json['image']
                b64 = raw.split(',', 1)[1] if ',' in raw else raw
                try:
                    file_bytes = base64.b64decode(b64)
                except Exception as ex:
                    return jsonify({'error': 'Base64 decode failed', 'detail': str(ex)}), 400
                filename = f"upload_{int(time.time())}.png"
                forwarded_path = FORWARD_DIR / filename
                with open(forwarded_path, "wb") as f:
                    f.write(file_bytes)
            else:
                raw_preview = request.get_data(as_text=True)[:300]
                return jsonify({'error': 'No file provided in form/files or JSON.', 'body_preview': raw_preview}), 400

        forward_result = None
        try:
            upload_url = KMEANS_HOST.rstrip("/") + "/upload"
            files = {"image": (forwarded_path.name, forwarded_path.read_bytes(), "image/png")}
            resp = requests.post(upload_url, files=files, timeout=20)
            try:
                j = resp.json()
            except Exception:
                j = {'status': resp.status_code, 'text': resp.text}
            forward_result = {'ok': 200 <= resp.status_code < 300, 'status_code': resp.status_code, 'response': j}
        except Exception as e:
            forward_result = {'ok': False, 'error': str(e)}

        public_url = f"/forwarded/{filename}"
        return jsonify({'saved_path': str(forwarded_path), 'filename': filename, 'public_url': public_url, 'forwarded_to_kmeans': forward_result}), 200

    except Exception as e:
        app.logger.exception("Upload failed")
        return jsonify({'error': 'Upload failed', 'detail': str(e)}), 500

# ---------------- 7. K-MEANS segmentation ----------------
@app.route('/kmeans', methods=['POST'])
def kmeans_segment():
    try:
        data_json = request.get_json(silent=True) or {}
        k = int(request.values.get('k') or data_json.get('k') or 4)
        resize_to = int(request.values.get('resize') or data_json.get('resize') or 256)

        filename = (request.values.get('filename') or data_json.get('filename'))
        pil_image = None

        if filename:
            filename = os.path.basename(filename)
            p = FORWARD_DIR / filename
            if not p.exists():
                return jsonify({'error': 'File not found on server', 'filename': filename}), 404
            pil_image = Image.open(str(p)).convert('RGB')
        else:
            if 'image' in request.files or 'file' in request.files:
                upload = request.files.get('image') or request.files.get('file')
                image_bytes = upload.read()
                pil_image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
            else:
                if data_json and 'image' in data_json:
                    raw = data_json['image']
                    b64 = raw.split(',', 1)[1] if ',' in raw else raw
                    try:
                        image_bytes = base64.b64decode(b64)
                    except Exception as ex:
                        return jsonify({'error': 'Base64 decode failed', 'detail': str(ex)}), 400
                    pil_image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
                else:
                    return jsonify({'error': 'No image provided (filename, upload or JSON).'}), 400

        orig_w, orig_h = pil_image.size
        pil_small = pil_image.resize((resize_to, resize_to), Image.LANCZOS)
        arr = np.array(pil_small).astype(np.float32)
        H, W = arr.shape[:2]
        flat = arr.reshape(-1, 3)
        km = KMeans(n_clusters=k, random_state=0, n_init=4)
        labels = km.fit_predict(flat)
        centers = km.cluster_centers_.astype(np.uint8)
        seg_flat = centers[labels]
        seg_img_small = seg_flat.reshape(H, W, 3)
        seg_pil_small = Image.fromarray(seg_img_small)
        seg_pil = seg_pil_small.resize((orig_w, orig_h), Image.NEAREST)

        safe_base = filename if filename else f"upload_{int(time.time())}.png"
        out_name = f"kmeans_{k}_" + os.path.basename(safe_base)
        out_path = FORWARD_DIR / out_name
        seg_pil.save(out_path)

        palette = [list(map(int, list(c))) for c in centers]
        public_url = f"/forwarded/{out_name}"
        return jsonify({'seg_public_url': public_url, 'out_name': out_name, 'saved_path': str(out_path), 'k': k, 'palette': palette}), 200

    except Exception as e:
        app.logger.exception("KMeans failed")
        return jsonify({'error': 'KMeans processing failed', 'detail': str(e)}), 500

# ---------------- 8. CLASSIFY (Beautiful bullet lists for UI) ----------------
@app.route('/classify', methods=['GET', 'POST'])
def classify_image():
    try:
        image_name = request.values.get('image') or request.values.get('filename')
        img = None

        if image_name:
            image_path = FORWARD_DIR / os.path.basename(image_name)
            if not image_path.exists():
                return jsonify({'error': 'Image not found on server.'}), 404
            img_bgr = cv2.imread(str(image_path))
            if img_bgr is None:
                return jsonify({'error': 'Failed to read image.'}), 500
            img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        else:
            if 'image' in request.files or 'file' in request.files:
                upload = request.files.get('image') or request.files.get('file')
                image_bytes = upload.read()
                img_pil = Image.open(io.BytesIO(image_bytes)).convert('RGB')
                img = np.array(img_pil)
            else:
                data_json = request.get_json(silent=True) or {}
                if data_json and 'image' in data_json:
                    raw = data_json['image']
                    b64 = raw.split(',', 1)[1] if ',' in raw else raw
                    try:
                        image_bytes = base64.b64decode(b64)
                    except Exception as ex:
                        return jsonify({'error': 'Base64 decode failed', 'detail': str(ex)}), 400
                    img_pil = Image.open(io.BytesIO(image_bytes)).convert('RGB')
                    img = np.array(img_pil)
                else:
                    return jsonify({'error': 'No image provided (filename, upload or JSON).'}), 400

        if img is None:
            return jsonify({'error': 'Failed to obtain image data.'}), 500

        rows, cols, bands = img.shape
        X = img.reshape(rows * cols, bands).astype(np.float32) / 255.0

        num_clusters = int(request.values.get('num_clusters') or request.values.get('k') or 5)
        kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
        kmeans.fit(X)
        labels = kmeans.labels_
        centers = kmeans.cluster_centers_
        classified = labels.reshape(rows, cols)

        total_pixels = rows * cols
        cluster_info = []
        cluster_sentences = []
        summary_items = []

        for i in range(num_clusters):
            mask = (classified == i)
            count = int(np.count_nonzero(mask))
            pct = round((count / total_pixels) * 100, 2)
            r, g, b = (centers[i] * 255)
            if b > 100 and b > g and b > r:
                label = "Likely Water (Blue Tone)"
            elif g > r and g > b:
                label = "Likely Vegetation (Green Tone)"
            elif r > 120 and g > 120 and b > 120:
                label = "Likely Urban / Built-up (Bright/Gray)"
            elif r > 100 and r > g and r > b:
                label = "Likely Soil / Barren Land (Reddish Tone)"
            else:
                label = "Mixed / Other Surface"

            cluster_info.append({
                'cluster_id': int(i),
                'label': label,
                'mean_rgb': (int(r), int(g), int(b)),
                'area_percent': pct
            })

            cluster_sentence = (
                f"<li style='margin-bottom:14px;'>"
                f"<strong>Cluster {i}: {label}</strong><br>"
                f"Mean RGB: ({int(r)}, {int(g)}, {int(b)})<br>"
                f"Area: {pct}% of image"
                f"</li>"
            )
            cluster_sentences.append(cluster_sentence)
            summary_items.append(f"{label} → {pct}%")

        out_filename = f"classified_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        out_path = OUTPUT_DIR / out_filename
        plt.imsave(str(out_path), classified, cmap='tab20', vmin=0, vmax=num_clusters-1)
        plt.close()
        output_url = f"/static/outputs/{out_filename}"

        cluster_bullets_html = (
            "<div style='max-width:720px;margin:0 auto;'>"
            "<h2 style='text-align:center;margin-bottom:18px;'>Cluster Analysis:</h2>"
            "<ul style='list-style:disc;padding-left:20px;'>"
            + "\n".join(cluster_sentences) +
            "</ul></div>"
        )

        summary_bullets_html = (
            "<div style='max-width:720px;margin:24px auto 0 auto;'>"
            "<h3 style='text-align:center;margin-bottom:12px;'>Land Cover Summary:</h3>"
            "<ul style='list-style:circle;padding-left:20px;'>"
            + "\n".join([f"<li>{s}</li>" for s in summary_items]) +
            "</ul></div>"
        )

        return jsonify({
            'output_url': output_url,
            'cluster_bullets': cluster_bullets_html,
            'summary_bullets': summary_bullets_html,
            'clusters': cluster_info,
        }), 200

    except Exception as e:
        app.logger.exception("Classification failed")
        return jsonify({'error': str(e)}), 500

# ---------------- 9. Serve forwarded files ----------------
@app.route('/forwarded/<path:fname>')
def forwarded_file(fname):
    p = FORWARD_DIR / fname
    if not p.exists():
        return "Not found", 404
    return send_from_directory(str(FORWARD_DIR.resolve()), fname)

# ---------------- 10. Serve outputs ----------------
@app.route('/static/outputs/<path:fname>')
def outputs_file(fname):
    p = OUTPUT_DIR / fname
    if not p.exists():
        return "Not found", 404
    return send_from_directory(str(OUTPUT_DIR.resolve()), fname)

# ---------------- Helper: strip HTML to plain text ----------------
def _html_to_plain(html: str) -> str:
    txt = re.sub(r'<br\s*/?>', '\n', html, flags=re.I)
    txt = re.sub(r'</li\s*>', '\n', txt, flags=re.I)
    txt = re.sub(r'<[^>]+>', '', txt)
    txt = txt.replace('&nbsp;', ' ')
    txt = txt.strip()
    return txt

# ---------------- Suggestion generator ----------------
def generate_suggestions(centers, percents):
    k = len(percents)
    labeled = []
    for i in range(k):
        r, g, b = (centers[i] * 255).astype(int)
        pct = round(percents[i], 2)
        if b > 100 and b > g and b > r:
            label = "water"
        elif g > r and g > b:
            label = "vegetation"
        elif r > 120 and g > 120 and b > 120:
            label = "urban/built-up"
        elif r > 100 and r > g and r > b:
            label = "soil/barren"
        else:
            label = "mixed/other"
        labeled.append((i, label, pct, (r,g,b)))

    labeled_sorted = sorted(labeled, key=lambda x: x[2], reverse=True)
    top = labeled_sorted[0]
    sec = labeled_sorted[1] if len(labeled_sorted) > 1 else None

    sentences = []
    s1 = f"The analysis shows dominant land cover is {top[1]} ({top[2]}%)."
    sentences.append(s1)
    if top[1] == "water":
        s2 = "Significant water presence suggests focusing on hydrological monitoring and flood risk assessment for the area."
    elif top[1] == "vegetation":
        s2 = "Strong vegetation signal suggests this area is suitable for agricultural monitoring — consider crop health and biomass monitoring."
    elif top[1] == "urban/built-up":
        s2 = "High urban/built-up fraction indicates potential zones for infrastructure assessment and urban expansion monitoring."
    elif top[1] == "soil/barren":
        s2 = "Large soil/barren area suggests land degradation or exposed surfaces; consider erosion control or land restoration planning."
    else:
        s2 = "The area contains mixed surfaces; further targeted analysis (e.g., higher-resolution imagery) is recommended to refine land-cover details."
    sentences.append(s2)

    if sec:
        if sec[1] != top[1]:
            s3 = f"The second-largest class is {sec[1]} ({sec[2]}%), so combined management strategies addressing both {top[1]} and {sec[1]} are advisable."
        else:
            s3 = "Secondary clusters show distributed variations; using time-series imagery would help confirm seasonal changes."
    else:
        s3 = "Consider acquiring more samples or time-series data to strengthen statistical confidence."
    sentences.append(s3)

    s4 = "Recommended actions: collect ground-truth samples where possible, prioritize monitoring based on dominant class, and re-run analysis after incorporating higher-resolution data or additional bands."
    sentences.append(s4)

    if top[2] > 60:
        final = " ".join(sentences[:3])
    else:
        final = " ".join(sentences[:4])

    return final

# ---------------- TWO-PAGE PDF generator using ReportLab ----------------
@app.route('/generate_pdf')
def generate_pdf():
    """
    Generates two-page PDF using ReportLab so images always embed reliably.
    Query params:
      - file=<colorized_filename> (required)
      - k=<num_clusters> (optional, default 4)
      - orig=<original_filename> (optional) — if provided use that path in static/uploads/
    """
    filename = request.args.get('file') or request.args.get('filename')
    if not filename:
        return jsonify({'error': 'Missing file parameter (file=...).'}), 400
    filename = os.path.basename(filename)
    k = int(request.args.get('k') or 4)
    orig_param = request.args.get('orig')  # optional: explicit original filename

    colorized_path = FORWARD_DIR / filename
    if not colorized_path.exists():
        return jsonify({'error': f'Colorized file not found: {filename}'}), 404

    # determine original file path:
    original_path = None
    if orig_param:
        p = UPLOAD_DIR / os.path.basename(orig_param)
        if p.exists():
            original_path = p

    # if not provided, try to find by timestamp match
    if original_path is None:
        digit_groups = re.findall(r'\d+', filename)
        ts_candidate = digit_groups[-1] if digit_groups else None
        if ts_candidate:
            for f in sorted(UPLOAD_DIR.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True):
                if not f.is_file():
                    continue
                if ts_candidate in f.name:
                    original_path = f
                    break

    # create kmeans image (ensuring RGB)
    kmeans_path = None
    centers_for_suggestions = None
    percents_for_suggestions = None
    centers_full = None
    labels_full = None
    try:
        with Image.open(str(colorized_path)) as img:
            img = img.convert('RGB')
            w, h = img.size
            small = img.resize((512, 512))
            arr = np.array(small).reshape(-1, 3).astype(np.float32)
            km_local = KMeans(n_clusters=k, random_state=0, n_init=4)
            labels_local = km_local.fit_predict(arr)
            centers_local = km_local.cluster_centers_.astype(np.uint8)
            seg = centers_local[labels_local].reshape(512, 512, 3)
            seg = Image.fromarray(seg).resize((w, h), Image.NEAREST).convert('RGB')
            kmeans_name = f"kmeans_{k}_{filename}"
            kmeans_path = FORWARD_DIR / kmeans_name
            seg.save(kmeans_path, format='PNG')

            centers_for_suggestions = centers_local.astype(np.float32) / 255.0
            counts = np.bincount(labels_local, minlength=k).astype(float)
            percents_for_suggestions = (counts / counts.sum() * 100.0).tolist()
    except Exception as e:
        app.logger.exception("KMeans (PDF gen) failed; continuing without kmeans image: %s", e)
        kmeans_path = None

    # classification to produce cluster lines
    cluster_lines = []
    summary_lines = []
    cluster_info = []
    try:
        arr_full = np.array(Image.open(str(colorized_path)).convert('RGB'))
        rows, cols, _ = arr_full.shape
        X_full = arr_full.reshape(-1, 3).astype(np.float32) / 255.0
        km_full = KMeans(n_clusters=k, random_state=42, n_init=10)
        km_full.fit(X_full)
        labels_full = km_full.labels_
        centers_full = km_full.cluster_centers_
        classified_full = labels_full.reshape(rows, cols)
        total_pixels = rows * cols

        for i in range(k):
            mask = (classified_full == i)
            count = int(np.count_nonzero(mask))
            pct = round((count / total_pixels) * 100, 2)
            r_f, g_f, b_f = (centers_full[i] * 255)
            r_i, g_i, b_i = int(r_f), int(g_f), int(b_f)
            if b_f > 100 and b_f > g_f and b_f > r_f:
                label = "Likely Water (Blue Tone)"
            elif g_f > r_f and g_f > b_f:
                label = "Likely Vegetation (Green Tone)"
            elif r_f > 120 and g_f > 120 and b_f > 120:
                label = "Likely Urban / Built-up (Bright/Gray)"
            elif r_f > 100 and r_f > g_f and r_f > b_f:
                label = "Likely Soil / Barren Land (Reddish Tone)"
            else:
                label = "Mixed / Other Surface"

            cluster_line = f"Cluster {i}: {label} | Mean RGB: ({r_i}, {g_i}, {b_i}) | Area: {pct}%"
            cluster_lines.append(cluster_line)
            summary_lines.append(f"{label} → {pct}%")
            cluster_info.append({'cluster_id': i, 'label': label, 'mean_rgb': (r_i, g_i, b_i), 'area_percent': pct})
    except Exception as e:
        app.logger.exception("Full classification (PDF) failed: %s", e)
        cluster_lines = ["Classification could not be generated."]
        summary_lines = []

    # compute SAR classifier confidence (prefer saved original if exists)
    sar_confidence = None
    try:
        if original_path and original_path.exists():
            ok, conf = is_sar_image(str(original_path))
            sar_confidence = conf
        else:
            with open(str(colorized_path), "rb") as f:
                b = f.read()
            okb, confb = is_sar_image_bytes(b)
            sar_confidence = confb
    except Exception:
        sar_confidence = None

    # suggestions paragraph
    suggestions_para = ""
    try:
        if centers_for_suggestions is not None and percents_for_suggestions is not None:
            suggestions_para = generate_suggestions(centers_for_suggestions, percents_for_suggestions)
        else:
            if centers_full is not None:
                centers_norm = (centers_full.astype(np.float32) / 255.0)
                counts2 = np.array([np.sum(labels_full == i) for i in range(len(centers_full))]).astype(float)
                perc2 = (counts2 / counts2.sum() * 100.0).tolist()
                suggestions_para = generate_suggestions(centers_norm, perc2)
            else:
                suggestions_para = "No suggestions available."
    except Exception:
        suggestions_para = "No suggestions available."

    # Now create PDF via ReportLab for robust image embedding
    try:
        from reportlab.pdfgen import canvas
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.utils import ImageReader
    except Exception as e:
        msg = "ReportLab is not installed. Install it: pip install reportlab"
        app.logger.exception(msg)
        return jsonify({'error': msg}), 500

    # Page sizes and layout (points)
    PAGE_W, PAGE_H = A4  # (595.2755905511812, 841.8897637795277)
    margin_pt = 36
    gap_pt = 12
    usable_w = PAGE_W - 2*margin_pt
    slot_w_pt = (usable_w - 2*gap_pt) / 3.0
    slot_h_pt = PAGE_H * 0.36

    pdf_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    pdf_name = f"analysis_report_{pdf_ts}.pdf"
    pdf_path = OUTPUT_DIR / pdf_name

    c = canvas.Canvas(str(pdf_path), pagesize=A4)

    # PAGE 1: Title
    title = "SAR Image Analysis Report"
    c.setFont("Helvetica-Bold", 18)
    c.drawCentredString(PAGE_W/2.0, PAGE_H - 36, title)

    # Helper: draw image into slot (keeps aspect)
    def draw_image_slot(cobj, pil_path, x_pt, y_pt, w_pt, h_pt, caption):
        try:
            with Image.open(str(pil_path)) as im:
                im = im.convert('RGB')
                iw, ih = im.size
                scale = min(w_pt / iw, h_pt / ih, 1.0)
                nw, nh = int(iw * scale), int(ih * scale)
                # Save a temporary JPEG in-memory for ImageReader (reportlab handles PNG but using JPEG is fine)
                buf = io.BytesIO()
                im.resize((nw, nh), Image.LANCZOS).save(buf, format='JPEG', quality=90)
                buf.seek(0)
                ir = ImageReader(buf)
                # center inside slot
                dx = x_pt + (w_pt - nw) / 2.0
                dy = y_pt + (h_pt - nh) / 2.0
                cobj.drawImage(ir, dx, dy, width=nw, height=nh)
                # caption under slot
                cobj.setFont("Helvetica", 10)
                cobj.drawCentredString(x_pt + w_pt/2.0, y_pt - 12, caption)
        except Exception as e:
            app.logger.exception("draw_image_slot failed for %s: %s", pil_path, e)
            # draw empty rectangle + caption
            cobj.rect(x_pt, y_pt, w_pt, h_pt)
            cobj.setFont("Helvetica", 10)
            cobj.drawCentredString(x_pt + w_pt/2.0, y_pt - 12, caption)

    # positions
    y_top = PAGE_H - 72 - slot_h_pt  # slot_y coordinate
    x0 = margin_pt
    # Slot 0: Original (from uploads or fallback to colorized)
    if original_path and original_path.exists():
        draw_image_slot(c, original_path, x0, y_top, slot_w_pt, slot_h_pt, "Original SAR")
    else:
        # fallback: colorized preview shown if original not found
        draw_image_slot(c, colorized_path, x0, y_top, slot_w_pt, slot_h_pt, "Original SAR (not found; preview shown)")

    # Slot 1: Colorized
    draw_image_slot(c, colorized_path, x0 + slot_w_pt + gap_pt, y_top, slot_w_pt, slot_h_pt, "Colorized")

    # Slot 2: KMeans
    if kmeans_path and kmeans_path.exists():
        draw_image_slot(c, kmeans_path, x0 + 2*(slot_w_pt + gap_pt), y_top, slot_w_pt, slot_h_pt, "K-Means Segmented")
    else:
        draw_image_slot(c, colorized_path, x0 + 2*(slot_w_pt + gap_pt), y_top, slot_w_pt, slot_h_pt, "K-Means (not available)")

    # Metadata block under images
    meta_x = margin_pt
    meta_y = y_top - 60
    line_h = 14
    c.setFont("Helvetica-Bold", 12)
    c.drawString(meta_x, meta_y, "Metadata:")
    c.setFont("Helvetica", 10)
    my = meta_y - 18
    original_name = original_path.name if original_path and original_path.exists() else "Not available"
    meta_items = [
        ("Original File", original_name),
        ("Colorized File", filename),
        ("Processed On", datetime.now().strftime("%Y-%m-%d %H:%M:%S IST")),
        ("SAR Classifier Confidence", f"{sar_confidence:.3f}" if sar_confidence is not None else "N/A"),
        ("Colorization Model", "Pix2Pix GAN"),
        ("KMeans Clusters", str(k))
    ]
    try:
        with Image.open(str(colorized_path)) as ti:
            meta_items.append(("Colorized Resolution", f"{ti.size[0]} x {ti.size[1]} px"))
    except Exception:
        meta_items.append(("Colorized Resolution", "N/A"))

    for key, val in meta_items:
        c.drawString(meta_x + 6, my, f"• {key}: {val}")
        my -= line_h

    # ---------------- FIXED LAYOUT: Summary BELOW Cluster Analysis ----------------

    # Draw Cluster Analysis first (full-width)
    col_x = margin_pt
    col_y = my - 8

    c.setFont("Helvetica-Bold", 12)
    c.drawString(col_x, col_y, "Cluster Analysis:")
    col_y -= 16
    c.setFont("Helvetica", 10)

    for cl in cluster_lines:
        wrapped = textwrap.wrap(cl, width=100)
        for ln in wrapped:
            c.drawString(col_x + 6, col_y, ln)
            col_y -= 12
        col_y -= 6

    # Now draw Land Cover Summary BELOW Cluster Analysis
    col_y -= 20
    c.setFont("Helvetica-Bold", 12)
    c.drawString(col_x, col_y, "Land Cover Summary:")
    col_y -= 16
    c.setFont("Helvetica", 10)

    for s in summary_lines:
        wrapped = textwrap.wrap(s, width=100)
        for ln in wrapped:
            c.drawString(col_x + 6, col_y, "• " + ln)
            col_y -= 12
        col_y -= 4

    # footer note
    footer_text = "Note: Colorized image is GAN-generated and for visualization only."
    c.setFont("Helvetica-Oblique", 8)
    c.drawCentredString(PAGE_W/2.0, 28, footer_text)

    c.showPage()

    # PAGE 2: Interpretation, Recommendations, Confidence, Limitations, Final Summary
    c.setFont("Helvetica-Bold", 16)
    c.drawCentredString(PAGE_W/2.0, PAGE_H - 36, "Analysis Details & Recommendations")
    y = PAGE_H - 80
    left = margin_pt
    c.setFont("Helvetica-Bold", 12)
    c.drawString(left, y, "Interpretation Notes:")
    y -= 18
    c.setFont("Helvetica", 10)
    interp_lines = []
    try:
        type_counts = {}
        for cinfo in cluster_info:
            lab = cinfo['label']
            type_counts[lab] = type_counts.get(lab, 0.0) + float(cinfo['area_percent'])
        sorted_types = sorted(type_counts.items(), key=lambda x: x[1], reverse=True)
        if sorted_types:
            top_label, top_pct = sorted_types[0]
            interp_lines.append(f"The area is dominated by {top_label.lower()} ({round(top_pct,2)}% combined).")
            for t, p in sorted_types[1:3]:
                interp_lines.append(f"Secondary presence: {t.lower()} ({round(p,2)}%).")
        else:
            interp_lines.append("No clear dominant land-cover type detected.")
        interp_lines.append("K-Means groups pixels by color similarity — labels are heuristic.")
        interp_lines.append("Consider adding multispectral or time-series data for improved analysis.")
    except Exception:
        interp_lines = ["Interpretation notes could not be automatically generated."]

    for ln in interp_lines:
        for wln in textwrap.wrap(ln, width=100):
            c.drawString(left, y, wln)
            y -= 14
        y -= 6

    y -= 6
    c.setFont("Helvetica-Bold", 12)
    c.drawString(left, y, "Recommendations:")
    y -= 18
    c.setFont("Helvetica", 10)
    for ln in textwrap.wrap(suggestions_para, width=120):
        c.drawString(left, y, ln)
        y -= 14
    y -= 8

    c.setFont("Helvetica-Bold", 12)
    c.drawString(left, y, "Confidence Indicators:")
    y -= 16
    c.setFont("Helvetica", 10)
    try:
        sar_text = f"SAR Detection Confidence: {sar_confidence:.3f}" if sar_confidence is not None else "SAR Detection Confidence: N/A"
        dominant_pct = max([c['area_percent'] for c in cluster_info]) if cluster_info else None
        dominant_text = f"Dominant Class Percentage: {dominant_pct:.2f}%" if dominant_pct is not None else "Dominant Class Percentage: N/A"
        reliability = "High" if (dominant_pct is not None and dominant_pct > 30) else ("Moderate" if dominant_pct and dominant_pct>15 else "Low")
        c.drawString(left, y, sar_text); y -= 14
        c.drawString(left, y, dominant_text); y -= 14
        c.drawString(left, y, f"Classification Reliability: {reliability}"); y -= 20
    except Exception:
        c.drawString(left, y, "Confidence data not available."); y -= 20

    c.setFont("Helvetica-Bold", 12)
    c.drawString(left, y, "Limitations:")
    y -= 16
    c.setFont("Helvetica", 10)
    limitations = [
        "KMeans is unsupervised and may not match exact land-cover classes.",
        "GAN colorization is synthetic and does not represent true spectral values.",
        "Classification depends on image quality and may contain uncertainty.",
        "Single-image analysis; use multi-temporal data for trend-based conclusions."
    ]
    for ln in limitations:
        for wln in textwrap.wrap(ln, width=110):
            c.drawString(left+6, y, "• " + wln)
            y -= 14
        y -= 6

    y -= 6
    c.setFont("Helvetica-Bold", 12)
    c.drawString(left, y, "Final Summary:")
    y -= 16
    c.setFont("Helvetica", 10)
    final_summary = [
        "The analysis indicates predominant vegetation with mixed surfaces.",
        "No major water bodies or dense urban zones detected in this image.",
        "Recommend periodic monitoring and ground-truth sampling to improve confidence."
    ]
    for ln in final_summary:
        for wln in textwrap.wrap(ln, width=110):
            c.drawString(left+6, y, wln)
            y -= 14
        y -= 6

    # Footer with filename & timestamp
    footer_left = f"File: {filename}"
    footer_right = datetime.now().strftime("%Y-%m-%d %H:%M:%S IST")
    c.setFont("Helvetica-Oblique", 8)
    c.drawString(margin_pt, 18, footer_left)
    c.drawRightString(PAGE_W - margin_pt, 18, footer_right)

    c.showPage()
    c.save()

    return send_from_directory(str(OUTPUT_DIR.resolve()), pdf_name, as_attachment=True)

# ---------------- Run App ----------------
if __name__ == '__main__':
    app.run(port=5010, debug=True, threaded=True)
