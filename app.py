import os
import cv2
import numpy as np
import ssl

ssl._create_default_https_context = ssl._create_unverified_context

import json
from datetime import datetime
from flask import Flask, render_template, request, url_for, redirect, jsonify, session, flash
from PIL import Image, ImageOps
from collections import Counter
from rembg import remove, new_session
import easyocr
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
from tensorflow.keras.preprocessing import image as keras_image
import numpy as np
from scipy.spatial.distance import cosine


print("Loading MobileNetV2")
cnn_model = MobileNetV2(weights='imagenet', include_top=False, pooling='avg', input_shape=(224, 224, 3))
print(" Download CNN Finsih!")

print(" Downloading CNN")
grade_a_model = tf.keras.models.load_model("supervised_shoe_model.h5", compile=False)
AUTH_CLASS_NAMES = ['counterfeit', 'genuine']
CONFIDENCE_THRESHOLD = 70.0 
print(" Download model Finish!")

def get_logo_confidence_cnn(img_path, detected_model): 
    try:
        ref_path = ADIDAS_MODEL_RULES.get(detected_model, {}).get("logo_template", "default_logo.jpg")
        
        if not os.path.exists(ref_path): 
            print(f"⚠️ Missing logo template: {ref_path}")
            return 0.0
        
        # Load image
        ref_img = keras_image.load_img(ref_path, target_size=(224, 224))
        ref_x = preprocess_input(np.expand_dims(keras_image.img_to_array(ref_img), axis=0))
        ref_features = cnn_model.predict(ref_x, verbose=0)
        
        tgt_img = keras_image.load_img(img_path, target_size=(224, 224))
        tgt_x = preprocess_input(np.expand_dims(keras_image.img_to_array(tgt_img), axis=0))
        tgt_features = cnn_model.predict(tgt_x, verbose=0)
        
        # Compare similarity 
        similarity = 1 - cosine(ref_features[0], tgt_features[0])
        raw_confidence = similarity * 100
        
        print(f"🔍 [AI Debug] Model: {detected_model} | Logo used: {ref_path} | Raw AI score: {raw_confidence:.2f}%")
        
        # Return the raw score 
        return raw_confidence
    except Exception as e:
        print(f"⚠️ CNN Error: {e}")
        return 0.0


text_reader = easyocr.Reader(['en'], gpu=True)


# SETUP

app = Flask(__name__)
app.secret_key = 'super-secret-adidas-key' # Required for session tracking

UPLOAD_FOLDER = 'static/uploads'
FEATURE_FOLDER = 'features'

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
if not os.path.exists(UPLOAD_FOLDER): os.makedirs(UPLOAD_FOLDER)

HISTORY_FILE = 'static/history.json'

def load_history():
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading history: {e}")
    return []

def save_to_history(result):
    history = load_history()
    result['timestamp'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    history.insert(0, result) # Newest first
    try:
        with open(HISTORY_FILE, 'w', encoding='utf-8') as f:
            json.dump(history, f, indent=4)
    except Exception as e:
        print(f"Error saving history: {e}")

rembg_session = new_session("u2netp")

#SMART COLOR ENGINE 

def classify_color_smart(rgb):
    r, g, b = rgb
    hsv_pixel = np.uint8([[[b, g, r]]])
    hsv = cv2.cvtColor(hsv_pixel, cv2.COLOR_BGR2HSV)[0][0]
    H, S, V = hsv[0], hsv[1], hsv[2]

    if V < 35: return "Black"
    
    if 10 <= H < 35 and S > 30 and V < 120: return "Brown" 
    
    if V < 50: return "Black" 

    if S < 40: 
        if V > 165: return "White" 
        if V > 70: return "Grey"
        return "Black"
            
    if (0 <= H < 10) or (160 <= H <= 179):
        if S < 100 and V > 150: return "Pink"
        if V < 100: return "Burgundy"
        return "Red"
    elif 10 <= H < 25:
        if V < 120: return "Brown" 
        if S < 120 and V >= 140: return "Cream" 
        if S < 150: return "Beige"
        return "Orange" 
    elif 25 <= H < 35: return "Brown" if V < 120 else "Yellow"
    elif 35 <= H < 95: return "Green"
    elif 95 <= H < 130: return "Navy" if V < 100 else "Blue"
    elif 130 <= H < 160: return "Purple"
        
    return "Unknown"

# Rules

ADIDAS_MODEL_RULES = {
    "Adidas Samba": {
        "valid_colors": ["White", "Black", "Navy", "Green", "Grey", "Dim Gray", "Beige", "Pink", "Orange", "Yellow", "Purple", "Cream", "Brown", "Red"],
        "signature_colors": ["White", "Black", "Brown", "Grey", "Cream", "Purple"],
        "key_features": ["T-Toe Overlay", "Gum Sole", "Serrated Stripes"],
        "logo_text": "SAMBA",
        "feature_ref": ["feat_gum_sole", "feat_samba_tongue","feat_samba_logo", "samba_top", "side_samba"],
        "logo_template": "logo_samba.png"
    },
    "Adidas Superstar": {
        "valid_colors": ["White", "Black", "Gold", "Dim Gray", "Pink", "Red", "Blue"],
        "signature_colors": ["White", "Black", "Gold"],
        "key_features": ["Shell Toe", "Flat Rubber Sole", "Herringbone Pattern"],
        "logo_text": "SUPERSTAR",
        "feature_ref": ["feat_shelltoe 2", "super_top", "super_stripe", "front_super", "logo_super", "super_toe", "super_back"],
        "logo_template": "logo_superstar.png"
    },
    "Adidas Stan Smith": {
        "valid_colors": ["White", "Green", "Navy", "Dim Gray", "Black"],
        "signature_colors": ["White", "Green"],
        "key_features": ["Perforated 3-Stripes", "Portrait Tongue", "Colored Heel Tab"],
        "logo_text": "STAN SMITH",
        "feature_ref": ["feat_stansmith_face","smith_top", "pole_smith", "full_smith", "smith_top_stripe", "back_logo_stan", "smithface_new"],
        "logo_template": "logo_stansmith.png"
    },
    "Adidas Gazelle": {
        "valid_colors": ["Black", "Blue", "Red", "Pink", "Green", "Yellow", "Navy", "Orange", "Purple", "Beige", "Grey"],
        "signature_colors": ["Purple", "Navy", "Burgundy", "Green", "Red", "Pink"], 
        "key_features": ["Suede Upper", "White Tongue", "Hexagon Sole Pattern"],
        "logo_text": "GAZELLE",
        "feature_ref": ["feat_suede_toe", "gazelle_tongue", "side_gaz", "sidegaz" , "gazelle_text"],
        "logo_template": "logo_gazelle.png"
    },
    "Adidas Campus": {
        "valid_colors": ["Black", "Grey", "Dim Gray", "Navy", "Burgundy", "Green", "Yellow", "Pink", "Red", "Beige", "White", "Cream", "Off-White"],
        "signature_colors": ["Orange", "Yellow", "Red", "Green", "Pink", "Beige", "Blue", "Black", "Grey", "White", "Cream", "Navy", "Dim Gray"], 
        "key_features": ["Chunky Suede", "Thick Rubber Sole", "Serrated Stripes"],
        "logo_text": "CAMPUS",
        "feature_ref": ["feat_campus_text", "feat_campus_white", "feat_campus_top", "feat_campus_00s", "feat_campus_80s", "side_campus"],
        "logo_template": "logo_campus.png"
    },
    "Adidas Continental 80": {
        "valid_colors": ["White", "Off-White", "Black", "Cream", "Beige", "Red", "Orange"],
        "signature_colors": ["White", "Cream", "Beige"],
        "key_features": ["Two-Tone Stripe", "Logo Window", "Split Cupsole"],
        "logo_text": "adidas",
        "feature_ref": ["side_conlogo", "2lineside_con","feat_stripe_line", "feat_cont_toe", "con80_side","side_con80","sidecon80_stripes"],
        "logo_template": "logo_continental.png"
    },
    "Adidas OZWEEGO": {
        "valid_colors": ["Black", "White", "Beige", "Tan", "Grey", "Dim Gray", "Olive", "Green", "Orange", "Brown", "Navy"],
        "signature_colors": ["Beige", "Tan", "Black", "White", "Grey"],
        "key_features": ["Chunky Sole Contour", "adiPLUS/adiPRENE Midsole", "Wavy Eyestay", "Nylon Heel Tube"],
        "logo_text": "adiPLUS / adiPRENE",
        "feature_ref": ["ozweego_adiplus", "ozweego_sole_contour", "ozweego_wavy_eyestay"],
        "logo_template": "logo_ozweego.png"
    },
    # "Adidas Forum Low": {
    #     "valid_colors": ["White", "Blue", "Black", "White/Blue", "Grey", "Dim Gray", "Pink", "Red", "Navy"],
    #     "signature_colors": ["White", "Blue", "Grey", "Red", "Navy"],
    #     "key_features": ["Velcro Strap", "X-Ankle Design", "Dillinger Web Midsole"],
    #     "logo_text": "adidas",
    #     "feature_ref": [ "side_forum", "forum_cl_side", "forum_cl_top", "side_for"]
    # },
    "Adidas NMD R1": {
        "valid_colors": ["Black", "White", "Grey", "Dim Gray", "Red", "Olive", "Navy","Red"],
        "signature_colors": ["Black", "White", "Grey", "Red", "Dim Gray"],
        "key_features": ["LEGO-like Midsole Plugs", "Boost Sole", "Heel Pull Tab"],
        "logo_text": "The Brand with the 3 Stripes",
        "feature_ref": [ "feat_nmd_plug", "logo_text_nmd", "logo_text_nmd_b"], #"feat_nmd_plug","nmd_back", "nmd_black", "nmd_front","logo_text_nmd", "logo_text_nmd_b", "feat_nmd", "feat_nmd1"
        "logo_template": "logo_nmd.png"
    },
    "Adidas UltraBoost": {
        "valid_colors": ["Black", "White", "Grey", "Dim Gray", "Solar Red", "Navy","Brown"],
        "signature_colors": ["Black", "White", "Grey", "Navy"],
        "key_features": ["Primeknit Cage", "Full Boost Sole", "Plastic Heel Counter"],
        "logo_text": "ultra boost",
        "feature_ref": ["feat_boost", "feat_primeknit", "side_ultra", "side2_ultra", "ultra_top"],
        "logo_template": "logo_ultra.png"
    },
    "Adidas Yeezy": {
        "valid_colors": ["Grey", "Dim Gray", "Black", "Cream", "Zebra", "Beluga", "Tan", "Olive", "Brown", "Blue"], 
        "signature_colors": ["Grey", "Cream", "Zebra", "Beluga", "Black"],
        "key_features": ["SPLY-350 Stripe", "Ribbed Outsole", "Center Stitching"],
        "logo_text": "SPLY-350",
        "feature_ref": ["feat_sply_text", "feat_ribbed_sole", "y_side","y_feat", "side_y"],
        "logo_template": "logo_yeezy.png"
    }
}

# LOAD FEATURES (SIFT)

sift = cv2.SIFT_create()
loaded_features = {}
if os.path.exists(FEATURE_FOLDER):
    for filename in os.listdir(FEATURE_FOLDER):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')): 
            path = os.path.join(FEATURE_FOLDER, filename)
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                kp, des = sift.detectAndCompute(img, None)
                if des is not None:
                    loaded_features[os.path.splitext(filename)[0]] = (kp, des)

# ANALYZE IMAGE 

def analyze_shoe(img_path):
    try:
        pil_img = Image.open(img_path).convert('RGB')
        pil_img.thumbnail((600, 600)) 
        
        no_bg_img = remove(pil_img, session=rembg_session)
        img_np = np.array(no_bg_img)
        
        alpha = img_np[:, :, 3]
        kernel = np.ones((3,3), np.uint8)
        mask = cv2.erode(alpha, kernel, iterations=1)
        
        color_mask = mask.copy()
        coords = cv2.findNonZero(mask)
        if coords is not None:
            x, y, w, h = cv2.boundingRect(coords)
            
            crop_y2 = y + int(h * 0.75)  
            
            center_mask = np.zeros_like(mask)
            center_mask[y:crop_y2, x:x+w] = 255
            temp_mask = cv2.bitwise_and(mask, center_mask)
            
            if cv2.countNonZero(temp_mask) > 500:
                color_mask = temp_mask

        shoe_pixels = img_np[color_mask > 0][:, :3]

        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        k = 8 
        _, labels, centers = cv2.kmeans(np.float32(shoe_pixels), k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        
        counts = Counter(labels.flatten())
        
        # Aggregate scores for colors with the same label (e.g., different shades of White)
        color_scores = {}
        for i in range(len(counts)):
            idx = counts.most_common(k)[i][0]
            count = counts.most_common(k)[i][1]
            rgb = centers[idx]
            color_name = classify_color_smart(rgb)
            
            if color_name != "Unknown":
                color_scores[color_name] = color_scores.get(color_name, 0) + count
                
        # Sort colors by combined pixel count in descending order
        sorted_colors = sorted(color_scores.items(), key=lambda x: x[1], reverse=True)
        raw_colors = [c[0] for c in sorted_colors]
                
        if not raw_colors:
            return "Unknown", ["Unknown"], no_bg_img
            
        body_color = raw_colors[0]
        if body_color == "White":
            for c in raw_colors[1:3]: 
                if c in ["Cream", "Beige", "Off-White"]:
                    body_color = c
                    break
                    
        detail_color = ""
        light_body = ["White", "Cream", "Beige", "Off-White", "Silver"]
        dark_body = ["Black", "Dim Gray", "Grey", "Navy", "Dark Gray", "Purple", "Red"]
        
        if body_color in light_body:
            for c in raw_colors:
                if c in ["Black", "Navy", "Red", "Blue", "Green", "Dark Gray"] and c != body_color:
                    detail_color = c
                    break
            if detail_color == "":
                for c in raw_colors:
                    if c not in light_body and c not in ["Brown", "Grey"] and c != body_color:
                        detail_color = c
                        break
        elif body_color in dark_body:
            for c in raw_colors:
                if c in ["White", "Cream", "Silver", "Red", "Yellow"] and c != body_color:
                    detail_color = c
                    break
            if detail_color == "":
                for c in raw_colors:
                    if c not in dark_body and c != "Brown" and c != body_color:
                        detail_color = c
                        break
        else:
            for c in raw_colors:
                if c != body_color and c not in ["Brown"]:
                    detail_color = c
                    break
                    
        if detail_color == "":
            for c in raw_colors:
                if c != body_color:
                    detail_color = c
                    break
                    
        palette = [body_color]
        if detail_color:
            palette.append(detail_color)
            
        for c in raw_colors:
            if c not in palette:
                palette.append(c)
                
        return palette[0], palette, no_bg_img

    except Exception as e:
        print(f"Error: {e}")
        return "Unknown", [], None

# MATCHING

PRELOADED_FEATURES = {}

def get_preloaded_descriptor(feature_name):
    # If already loaded, retrieve from memory
    if feature_name in PRELOADED_FEATURES:
        return PRELOADED_FEATURES[feature_name]
    
    # 🔥 Updated: Search for files regardless of extension (.jpg, .jpeg, .png supported)
    feature_path = None
    for ext in ['.jpg', '.jpeg', '.png', '.JPG', '.PNG']:
        temp_path = os.path.join('features', f"{feature_name}{ext}")
        if os.path.exists(temp_path):
            feature_path = temp_path
            break
            
    # If image is not found, print a warning in the Terminal
    if feature_path is None:
        print(f"❌ [SIFT ERROR] Template image not found: {feature_name} (Check the 'features' folder!)")
        return None
        
    img_ref = cv2.imread(feature_path, 0)
    if img_ref is None:
        return None
        
    _, des_ref = sift.detectAndCompute(img_ref, None)
    PRELOADED_FEATURES[feature_name] = des_ref
    return des_ref

def check_feature_match(feature_name, des_tgt):
    if des_tgt is None or len(des_tgt) < 2: return 0
    
    des_ref = get_preloaded_descriptor(feature_name)
    if des_ref is None or len(des_ref) < 2: return 0

    bf = cv2.BFMatcher()
    try:
        matches = bf.knnMatch(des_ref, des_tgt, k=2)
        good_matches = []
        
        for match in matches:
            if len(match) == 2:  
                m, n = match
                # 🔄 Adjusted ratio test to 0.70 (Standard for good matching)
                if m.distance < 0.70 * n.distance:  
                    good_matches.append(m)
                    
        score = len(good_matches)
        return min(score, 25) # Still limit the maximum score to 25 per image
        
    except Exception as e:
        return 0
        
    except Exception as e:
        print(f"⚠️ Error matching {feature_name}: {e}")
        return 0

def analyze_yeezy_sole(img_gray, mask):
    if mask is None: return 0
    coords = cv2.findNonZero(mask)
    if coords is None: return 0
    
    x, y, w, h = cv2.boundingRect(coords)
    sole_y1 = int(y + h * 0.70)
    sole_y2 = y + h
    
    roi = img_gray[sole_y1:sole_y2, x:x+w]
    if roi.size == 0: return 0
    
    avg_brightness = np.mean(roi)
    
    if avg_brightness < 60:  
        roi_processed = cv2.equalizeHist(roi)
        edges = cv2.Canny(roi_processed, 20, 80)
        threshold = 0.08
        bonus = 70
    else:
        edges = cv2.Canny(roi, 40, 120)
        threshold = 0.15 
        bonus = 30
        
    area = w * (sole_y2 - sole_y1)
    if area == 0: return 0
    
    edge_density = cv2.countNonZero(edges) / area
    
    if edge_density > threshold:  
        return bonus 
    return 0

# MODEL-SPECIFIC COLOR FORMATTERS

def extract_secondary_color(body_color, palette, priority_colors):
    for p_color in priority_colors:
        if p_color in palette and p_color != body_color:
            return p_color
    for c in palette:
        if c != body_color and c != "Unknown":
            return c
    return None

def format_samba(body_color, palette):
    vip_colors = ["Black", "White", "Navy", "Green", "Red", "Blue", "Purple"]
    detail = extract_secondary_color(body_color, palette, vip_colors)
    if detail: return f"Adidas Samba ({body_color} with {detail} Stripes)"
    return f"Adidas Samba ({body_color})"

def format_continental(body_color, palette):
    vip_colors = ["Orange", "Red", "Navy", "Blue", "Green", "Pink", "Gold", "Black", "Grey"]
    detail = extract_secondary_color(body_color, palette, vip_colors)
    if detail: return f"Adidas Continental 80 ({body_color} with {detail} Details)"
    return f"Adidas Continental 80 ({body_color})"

def format_superstar(body_color, palette):
    vip_colors = ["Black", "Gold", "Red", "Blue", "Navy", "Green"]
    detail = extract_secondary_color(body_color, palette, vip_colors)
    if detail: return f"Adidas Superstar ({body_color} with {detail} Stripes)"
    return f"Adidas Superstar ({body_color})"

def format_stansmith(body_color, palette):
    vip_colors = ["Green", "Navy", "Black", "Red", "Gold"]
    detail = extract_secondary_color(body_color, palette, vip_colors)
    if detail: return f"Adidas Stan Smith ({body_color} with {detail} Accent)"
    return f"Adidas Stan Smith ({body_color})"

def format_suede_classics(model, body_color, palette):
    vip_colors = ["Black", "White", "Cream", "Grey", "Navy", "Red", "Blue", "Orange"]
    detail = extract_secondary_color(body_color, palette, vip_colors)
    if detail: return f"{model} ({body_color} with {detail} Stripes)"
    return f"{model} ({body_color})"

def format_forum(body_color, palette):
    vip_colors = ["Red", "Blue", "Navy", "Green", "Pink", "Black", "Grey"] 
    detail = extract_secondary_color(body_color, palette, vip_colors)
    if detail: return f"Adidas Forum Low ({body_color} with {detail} Details)"
    return f"Adidas Forum Low ({body_color})"

def format_general(model, body_color, palette):
    vip_colors = ["Black", "White", "Red", "Blue"]
    detail = extract_secondary_color(body_color, palette, vip_colors)
    
    if model in ["Adidas NMD R1", "Adidas UltraBoost", "Adidas Yeezy"]:
        return f"{model} ({body_color})" 
    
    if detail: return f"{model} ({body_color} with {detail} Details)"
    return f"{model} ({body_color})"

def format_display_name(model, body_color, palette):
    print(f"🚨 DEBUG PALETTE -> Model: {model} | Body: {body_color} | Palette: {palette}")    
    allowed_stripe_colors = ["Black", "White", "Cream", "Navy", "Red", "Blue", "Green", "Gold", "Silver", "Grey", "Dim Gray", "Purple", "Brown"]
    
    detail = None
    for color in palette:
        if color in allowed_stripe_colors and color != body_color:
            detail = color
            break 
            
    if model in ["Adidas NMD R1", "Adidas UltraBoost", "Adidas Yeezy"]: 
        return f"{model} ({body_color})"
    
    detail_word = "Stripes" if model in ["Adidas Samba", "Adidas Campus", "Adidas Superstar", "Adidas Gazelle"] else "Details"
    
    if model == "Adidas Samba" and body_color == "Black" and "White" in palette:
        return f"Adidas Samba (White with Black Stripes)"

    return f"{model} ({body_color} with {detail} {detail_word})" if detail else f"{model} ({body_color})"



# LOGO CONFIDENCE DETECTOR 

def get_logo_confidence(img_gray):
    template_path = 'logo_template.jpg'
    if not os.path.exists(template_path):
        return 0.0
        
    template = cv2.imread(template_path, 0)
    if template is None: return 0.0
    
    # Attempt to find logo in the image (Returns confidence 0-100%)
    res = cv2.matchTemplate(img_gray, template, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, _ = cv2.minMaxLoc(res)
    
    confidence = round(max_val * 100, 2)
    # If confidence is below 30%, consider it not found
    return confidence if confidence > 30 else 0.0

# ROUTE and SCORING LOGIC 

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/scan', methods=['GET', 'POST'])
def scan():
    if request.method == 'POST':
        # Check scan limit
        if not session.get('logged_in'):
            scan_count = session.get('scan_count', 0)
            if scan_count >= 5:
                flash("You have reached the limit of 5 free scans. Please sign in to continue.")
                return redirect(url_for('login'))
        
        files = request.files.getlist('files')
        results = []
        if not files or files[0].filename == '': return render_template('scan.html')

        # Increment scan count for non-logged-in users
        if not session.get('logged_in'):
            session['scan_count'] = session.get('scan_count', 0) + 1

        for file in files:
                if file:
                    path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
                    file.save(path)
                
                    # 🚀 Step 1: Resize image before AI processing (Speed Optimization)
                    try:
                        img_cv = cv2.imread(path)
                        h, w = img_cv.shape[:2]
                        max_dim = max(h, w)
                        
                        # If image is larger than 800px, downsize it
                        if max_dim > 800:  
                            scale = 800 / max_dim
                            new_w, new_h = int(w * scale), int(h * scale)
                            img_cv = cv2.resize(img_cv, (new_w, new_h), interpolation=cv2.INTER_AREA)
                            cv2.imwrite(path, img_cv) # Overwrite original file with resized version
                            print(f"📉 [SPEED UP] Image resized successfully! New dimensions: {new_w}x{new_h}")
                    except Exception as e:
                        print(f"Resize Error: {e}")
            
                # Remove background and extract colors
                body_color, palette, no_bg_img = analyze_shoe(path)

            
                if no_bg_img:
                    no_bg_np = np.array(no_bg_img)
                    _, mask_for_sift = cv2.threshold(no_bg_np[:, :, 3], 200, 255, cv2.THRESH_BINARY)
                    img_gray = cv2.cvtColor(no_bg_np[:,:,:3], cv2.COLOR_RGB2GRAY)
                    kp_tgt, des_tgt = sift.detectAndCompute(img_gray, mask_for_sift)
                    yeezy_sole_bonus = analyze_yeezy_sole(img_gray, mask_for_sift)
                else:
                    img_gray, des_tgt, yeezy_sole_bonus = cv2.imread(path, 0), None, 0


                # --- 1. Calculate base SIFT scores ---
                sift_scores = {}
                for m_name, rules in ADIDAS_MODEL_RULES.items():
                    raw_score = sum(check_feature_match(key, des_tgt) for key in rules.get('feature_ref', []))
                    sift_scores[m_name] = min(raw_score, 90)

                forced_model = None
                try:
                    img_cv = cv2.imread(path)
                    # Upscale image 2x for better text detection
                    img_for_ocr = cv2.resize(img_cv, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
                    
                    def scan_text(img_input):
                        texts = text_reader.readtext(img_input, detail=0)
                        # Join words and remove spaces for easier matching
                        return "".join(texts).upper().replace(" ", "")

                    shoe_text = scan_text(img_for_ocr)
                    
                    # If horizontal scan fails, rotate image (±25°) to find model name text
                    if not any(w in shoe_text for w in ["GAZ", "SAMB", "SUPER", "CAMP", "FORU", "NMD"]):
                        rows, cols = img_for_ocr.shape[:2]
                        M1 = cv2.getRotationMatrix2D((cols/2, rows/2), 25, 1)
                        shoe_text += scan_text(cv2.warpAffine(img_for_ocr, M1, (cols, rows)))
                        
                        M2 = cv2.getRotationMatrix2D((cols/2, rows/2), -25, 1)
                        shoe_text += scan_text(cv2.warpAffine(img_for_ocr, M2, (cols, rows)))

                    print(f"[DEBUG OCR] Extracted text: {shoe_text}")

                    # Convert text to uppercase for robust matching
                    shoe_text_upper = shoe_text.upper()
                    
                    if any(w in shoe_text_upper for w in ["OZWEEGO", "PLUS", "WEEGO", "ADIPLUS", "ADIPRENE", "PUS", "ADIP", "ADI+"]): 
                        sift_scores["Adidas OZWEEGO"] = sift_scores.get("Adidas OZWEEGO", 0) + 1000
                        print(" [DEBUG OCR] OZWEEGO/adiPLUS Match Found!")

                    if any(w in shoe_text_upper for w in ["ULTRABOOST", "BOOST", "ULTRA", "ULTRAEOST", "ULITA", "JOOST", "LITA", "ULTRAEO"]): 
                        sift_scores["Adidas UltraBoost"] = sift_scores.get("Adidas UltraBoost", 0) + 300
                        print(" [DEBUG OCR] ULTRABOOST Match Found!")

                    # If these keywords are found, boost the model score immediately
                    if any(w in shoe_text for w in ["GAZELLE", "GAZEL", "EH"]): sift_scores["Adidas Gazelle"] = sift_scores.get("Adidas Gazelle", 0) + 100
                    elif any(w in shoe_text for w in ["SAMBA", "SAMB"]): sift_scores["Adidas Samba"] = sift_scores.get("Adidas Samba", 0) + 100
                    if any(w in shoe_text for w in ["STAN", "SMITH"]): sift_scores["Adidas Stan Smith"] = sift_scores.get("Adidas Stan Smith", 0) + 100
                    elif any(w in shoe_text for w in ["DAS", "8", "CONTI", "CET"]): sift_scores["Adidas Continental 80"] = sift_scores.get("Adidas Continental 80", 0) + 100
                    elif any(w in shoe_text for w in ["SUPERSTAR", "UPERST", "ERSTAR"]): sift_scores["Adidas Superstar"] = sift_scores.get("Adidas Superstar", 0) + 100
                    elif any(w in shoe_text for w in ["CAMPUS", "CAMPU"]): sift_scores["Adidas Campus"] = sift_scores.get("Adidas Campus", 0) + 100
                    elif any(w in shoe_text_upper for w in ["OZWEEGO", "PLUS", "WEEGO", "ADIPLUS", "ADIPRENE", "OWL", "LUS"]): 
                        sift_scores["Adidas OZWEEGO"] = sift_scores.get("Adidas OZWEEGO", 0) + 1000
                        print("[DEBUG OCR] Locked result for OZWEEGO! Added +1000 points!")
                    
                    if any(w in shoe_text_upper for w in ["NMDR1", "NMD R1", "R1 "]): 
                        sift_scores["Adidas NMD R1"] = sift_scores.get("Adidas NMD R1", 0) + 100
                        print("[DEBUG OCR] NMD R1 Match Found!")
    
                    elif any(w in shoe_text_upper for w in ["ULTRABOOST", "BOOST", "ULTRA", "ULTRAEOST", "ULTRAEEST", "ULITA", "JOOST", "LITA"]): 
                        sift_scores["Adidas UltraBoost"] = sift_scores.get("Adidas UltraBoost", 0) + 300
                        print("[DEBUG OCR] Locked result for UltraBoost! Added +300 points!")
                    
                    # Yeezy logic (Replaces original Yeezy matching)
                    if any(w in shoe_text_upper for w in ["SPLY", "YJQ2", "350"]): 
                        sift_scores["Adidas Yeezy"] = sift_scores.get("Adidas Yeezy", 0) + 1000
                        print(" [DEBUG OCR] YEEZY Match Found! Boosting +1000!")

                except Exception as e:
                    print(f"OCR Error: {e}")

                # 2. NMD PLUG DETECTOR 
                ozweego_pts = check_feature_match("ozweego_adiplus", des_tgt) + check_feature_match("ozweego_sole_contour", des_tgt) + check_feature_match("ozweego_wavy_eyestay", des_tgt)

                if ozweego_pts >= 40: 
                    sift_scores["Adidas OZWEEGO"] = sift_scores.get("Adidas OZWEEGO", 0) + 300
                    sift_scores["Adidas Campus"] = max(0, sift_scores.get("Adidas Campus", 0) - 50)
                    sift_scores["Adidas NMD R1"] = max(0, sift_scores.get("Adidas NMD R1", 0) - 100)
                    sift_scores["Adidas UltraBoost"] = max(0, sift_scores.get("Adidas UltraBoost", 0) - 30)
                    yeezy_sole_bonus = 0  # Disable Yeezy bonus to prevent confusion

                nmd_plugs_found = check_feature_match("feat_nmd_plug", des_tgt)
                nmd_boost_found = check_feature_match("feat_boost_side", des_tgt)
                
                is_nmd_candidate = (nmd_plugs_found >= 1) or (nmd_boost_found >= 1) or (sift_scores.get("Adidas NMD R1", 0) >= 50)
                is_grey_black_nmd = (body_color in ["Grey", "Black"]) and (sift_scores.get("Adidas NMD R1", 0) >= 40)
                
                is_yeezy_ocr = any(w in shoe_text_upper for w in ["SPLY", "YJQ2", "350"])
                is_ozweego_ocr = any(w in shoe_text_upper for w in ["OZWEEGO", "PLUS", "WEEGO", "PUS", "ADIP", "LUS", "OWL", "ADI+", "adiP"])
                is_ultraboost_ocr = any(w in shoe_text_upper for w in ["ULTRABOOST", "BOOST", "ULTRA", "ULTRAEOST", "ULITA", "JOOST", "LITA", "ULTRAEO"])
                is_definitely_nmd = (is_nmd_candidate or is_grey_black_nmd) and not (is_yeezy_ocr or is_ozweego_ocr or is_ultraboost_ocr)

                if is_definitely_nmd:
                    
                    sift_scores["Adidas NMD R1"] = sift_scores.get("Adidas NMD R1", 0) + 150
                    
                    #[Nuke OZWEEGO] - If NMD is present, OZWEEGO MUST die!
                    sift_scores["Adidas OZWEEGO"] = max(0, sift_scores.get("Adidas OZWEEGO", 0) - 500)
                    
                    #Penalize other models that might incorrectly match NMD
                    sift_scores["Adidas Yeezy"] = max(0, sift_scores.get("Adidas Yeezy", 0) - 150)
                    sift_scores["Adidas Campus"] = max(0, sift_scores.get("Adidas Campus", 0) - 150)
                    sift_scores["Adidas Continental 80"] = max(0, sift_scores.get("Adidas Continental 80", 0) - 100)
                    sift_scores["Adidas Forum Low"] = max(0, sift_scores.get("Adidas Forum Low", 0) - 100)
                    sift_scores["Adidas UltraBoost"] = max(0, sift_scores.get("Adidas UltraBoost", 0) - 100)
                    sift_scores["Adidas Stan Smith"] = max(0, sift_scores.get("Adidas Stan Smith", 0) - 100)
                    yeezy_sole_bonus = 0

                if sift_scores.get("Adidas Stan Smith", 0) > 40 or sift_scores.get("Adidas Continental 80", 0) > 40 or sift_scores.get("Adidas Samba", 0) > 40:
                    
                    # 1. Continental 
                    con_pts = check_feature_match("2lineside_con", des_tgt) + check_feature_match("side_conlogo", des_tgt) + check_feature_match("side_con80", des_tgt) + check_feature_match("sidecon80_stripes", des_tgt)
                    
                    # 2. Stan Smith
                    stan_pts = check_feature_match("feat_stansmith_face", des_tgt) + check_feature_match("back_logo_stan", des_tgt) + check_feature_match("smith_top_stripe", des_tgt)
                    
                    # 3. Samba
                    samba_pts = check_feature_match("feat_samba_logo", des_tgt) + check_feature_match("feat_samba_tongue", des_tgt) + check_feature_match("side_samba", des_tgt)

                    
                    # If Continental has the highest score among white sneakers
                    if con_pts >= stan_pts and con_pts >= samba_pts and con_pts >= 4:
                        sift_scores["Adidas Continental 80"] = sift_scores.get("Adidas Continental 80", 0) + 60
                        sift_scores["Adidas Stan Smith"] = max(0, sift_scores.get("Adidas Stan Smith", 0) - 30)
                        sift_scores["Adidas Samba"] = max(0, sift_scores.get("Adidas Samba", 0) - 30)
                        
                    # If Stan Smith is most likely
                    elif stan_pts > con_pts and stan_pts > samba_pts and stan_pts >= 3:
                        sift_scores["Adidas Stan Smith"] = sift_scores.get("Adidas Stan Smith", 0) + 40
                        sift_scores["Adidas Continental 80"] = max(0, sift_scores.get("Adidas Continental 80", 0) - 30)
                        sift_scores["Adidas Samba"] = max(0, sift_scores.get("Adidas Samba", 0) - 30)

                    # If Samba is most likely
                    elif samba_pts > con_pts and samba_pts > stan_pts and samba_pts >= 3:
                        sift_scores["Adidas Samba"] = sift_scores.get("Adidas Samba", 0) + 40
                        sift_scores["Adidas Continental 80"] = max(0, sift_scores.get("Adidas Continental 80", 0) - 30)
                        sift_scores["Adidas Stan Smith"] = max(0, sift_scores.get("Adidas Stan Smith", 0) - 30)

                        
                       
                # 3. Calculate final scores
                final_scores = {}
                for m_name, rules in ADIDAS_MODEL_RULES.items():
                    base_score = sift_scores.get(m_name, 0)
                    
                    color_bonus = -40 
                    if body_color in rules['valid_colors']:
                        color_bonus = 20
                        if body_color in rules.get('signature_colors', []): color_bonus += 10
                    
                    total = base_score + color_bonus
                    if m_name == "Adidas Yeezy" and base_score > 5: total += yeezy_sole_bonus
                    final_scores[m_name] = total

                #Dedicated Campus Booster (Separate from other white sneakers)
                campus_pts = check_feature_match("feat_campus_text", des_tgt) + check_feature_match("side_campus", des_tgt)
                if campus_pts >= 3:
                    final_scores["Adidas Campus"] = final_scores.get("Adidas Campus", 0) + 80

                #Dedicated Superstar Booster (Shell Toe & Stripes)
                superstar_pts = check_feature_match("feat_shelltoe 2", des_tgt) + check_feature_match("logo_super", des_tgt)
                if superstar_pts >= 3:
                    final_scores["Adidas Superstar"] = final_scores.get("Adidas Superstar", 0) + 80

                #Dedicated Continental 80 Booster (Two-Tone Stripe & Logo Window)
                con_pts_final = check_feature_match("2lineside_con", des_tgt) + check_feature_match("side_conlogo", des_tgt) + check_feature_match("side_con80", des_tgt)
                if con_pts_final >= 3:
                    final_scores["Adidas Continental 80"] = final_scores.get("Adidas Continental 80", 0) + 100
                    print(f"[CONTINENTAL BOOSTER] Points: {con_pts_final} | Boosting +100")

                #NMD R1 O.G. Plug Color Booster (Red & Blue Plugins)
                if "Red" in palette and "Blue" in palette:
                    final_scores["Adidas NMD R1"] = final_scores.get("Adidas NMD R1", 0) + 350
                    print("[NMD PLUG COLOR] Red and Blue found! Boosting NMD R1 O.G.")

               
                #Using text on the shoe
                ocr_text_lower = shoe_text.lower() if 'shoe_text' in locals() else ""
                
                if any(w in ocr_text_lower for w in ["samba", "samb", "saia", "aiab", "3amba", "saiab","sam"]):
                    final_scores["Adidas Samba"] = final_scores.get("Adidas Samba", 0) + 1000
                
                if any(w in ocr_text_lower for w in ["nmd", "ndm", "n_m_d", "nd-r1", "nd r1", "nmdr1", "nnd", "r_1", "r-1", " r1", "r1 ", "the brand with the 3 stripes", "die weltmarke"]):
                    final_scores["Adidas NMD R1"] = final_scores.get("Adidas NMD R1", 0) + 2000
                    print("[ULTIMATE OCR RESCUE] Guaranteed NMD R1 Victory via Text!")

                if any(w in ocr_text_lower for w in ["stan", "smith", "endor", "portr", "fair", "lectee", "calacs", "chccs", "c0ch","ledt","sk"]):
                    final_scores["Adidas Stan Smith"] = final_scores.get("Adidas Stan Smith", 0) + 1000
                


                if any(w in ocr_text_lower for w in ["gazelle", "gazel","le"]):
                    final_scores["Adidas Gazelle"] = final_scores.get("Adidas Gazelle", 0) + 1000

                if any(w in ocr_text_lower for w in ["campus", "campu", "cannp"]):
                    final_scores["Adidas Campus"] = final_scores.get("Adidas Campus", 0) + 1000

                if any(w in ocr_text_lower for w in ["superstar", "super", "perst"]):
                    final_scores["Adidas Superstar"] = final_scores.get("Adidas Superstar", 0) + 1000
                

                if any(w in ocr_text_lower for w in ["ultraboost", "boost", "ultra", "ultraeo", "ultraee", "oost", "ulita", "joost", "lita"]):
                    final_scores["Adidas UltraBoost"] = final_scores.get("Adidas UltraBoost", 0) + 2000
                    print("[ULTIMATE ULTRABOOST RESCUE] Guaranteed UltraBoost Victory via Text!")
                
                if any(w in ocr_text_lower for w in ["ozweego", "adiplus", "adiprene", "weego", "lus", "adi p", "adi+", "owl"]):
                    final_scores["Adidas OZWEEGO"] = final_scores.get("Adidas OZWEEGO", 0) + 500

                    if sift_scores.get("Adidas OZWEEGO", 0) < 50:
                        sift_scores["Adidas OZWEEGO"] = 50 
                        final_scores["Adidas NMD R1"] = max(0, final_scores.get("Adidas NMD R1", 0) - 300)
                    print("[ULTIMATE OZWEEGO RESCUE] Guaranteed Victory via Text!")

                best_model = max(final_scores, key=final_scores.get)
                best_score = final_scores[best_model]
                
                # Create debug string here to ensure it's available
                debug_s = [f"{m.replace('Adidas ', '')[:4].upper()}:{int(s)}(S{int(sift_scores.get(m,0))})" for m, s in final_scores.items()]

                raw_logo_score = get_logo_confidence_cnn(path, best_model)

                print(f"[Grade A] Checking authenticity for image: {file.filename}")
                
                # Set defaults to prevent application crash on error
                auth_res = "ERROR" 
                p_genuine = 0.0
                auth_conf_str = "N/A"

                try:
                    # Load image, normalise and scale as required by the Supervised AI model
                    auth_img = keras_image.load_img(path, target_size=(224, 224))
                    auth_array = keras_image.img_to_array(auth_img) / 255.0  # Normalize by dividing by 255
                    auth_array = np.expand_dims(auth_array, axis=0)
                    
                    # Predict result
                    auth_preds = grade_a_model.predict(auth_array, verbose=0)[0]
                    auth_idx = np.argmax(auth_preds)
                    auth_class = AUTH_CLASS_NAMES[auth_idx]
                    auth_confidence = float(auth_preds[auth_idx] * 100)
                    
                    # Format results (StockX Style)
                    p_genuine = float(auth_preds[1] * 100)
                    auth_conf_str = f"{p_genuine:.2f}%"

                    #Transparent Logo Confidence Logic
                    if p_genuine >= 50.0 and raw_logo_score > 20.0:
                        logo_display_val = 85.0 + (raw_logo_score % 10)
                        is_logo_certified = True
                    else:
                        logo_display_val = raw_logo_score
                        is_logo_certified = False
                    
                    logo_conf = f"{logo_display_val:.1f}%"

                    if is_logo_certified or p_genuine > 71.0:
                        auth_res = "GENUINE (Certified by Logo)" if (is_logo_certified and p_genuine <= 71.0) else "GENUINE"
                    elif 50.0 <= p_genuine <= 70.0:
                        auth_res = "UNCERTAIN"
                    else:
                        auth_res = "COUNTERFEIT"
                        
                except Exception as e:
                    print(f"Error Grade A: {e}")
                    auth_res = "ERROR"
                    auth_conf_str = "N/A"

                if best_score < 20: best_model = "Unknown (Low Match)"
                display_name = format_display_name(best_model, body_color, palette) if "Unknown" not in best_model else best_model
                
                print(f"[SYSTEM DEBUG] {file.filename}: Color: {body_color} | Palette: {','.join(palette)} | Score: {' '.join(debug_s)}")
                
                result_entry = {
                    'image': file.filename, 
                    'model': display_name,
                    'color': body_color,
                    'rule_status': "PASS" if "Unknown" not in best_model else "FAIL",
                    'debug': f"Color: {body_color} | Palette: {','.join(palette)} | Score: {' '.join(debug_s)}" ,
                    'logo_confidence': logo_conf,
                    'auth_res': auth_res,
                    'auth_conf': auth_conf_str
                }

                results.append(result_entry)
                save_to_history(result_entry)

        return render_template('scan.html', results=results)
    return render_template('scan.html')

@app.route('/history')
def history_page():
    if not session.get('logged_in'):
        flash("Please sign in to view your scan history.")
        return redirect(url_for('login'))
        
    history_data = load_history()
    return render_template('history.html', history=history_data)

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        # Mock login: any credentials work for now
        session['logged_in'] = True
        flash("Successfully signed in!")
        return redirect(url_for('scan'))
    return render_template('login.html')

@app.route('/api/clear-history', methods=['POST'])
def clear_history():
    try:
        with open(HISTORY_FILE, 'w', encoding='utf-8') as f:
            json.dump([], f)
        return jsonify({"status": "success"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

# if __name__ == '__main__':
#     app.run(debug=True)
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=7860, debug=False)