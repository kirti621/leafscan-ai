from flask import Flask, request, jsonify, send_from_directory, render_template, redirect, session, url_for, flash
from flask_cors import CORS
from werkzeug.utils import secure_filename
from werkzeug.security import generate_password_hash, check_password_hash
from PIL import Image
import sqlite3
import json
import datetime
import os
import numpy as np

# ─── Lazy-load heavy deps so app starts even without GPU ───
try:
    import tensorflow as tf
    from tensorflow.keras.preprocessing import image as keras_image
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    print("⚠️  TensorFlow not found. Predictions will be disabled.")

# ═══════════════════════════════════════════════
#  APP SETUP
# ═══════════════════════════════════════════════
app = Flask(__name__)
app.secret_key = "leafscan_secret_2025_xK9#mP"
CORS(app)

UPLOAD_FOLDER = os.path.join("static", "uploads")
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "webp", "bmp"}
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ─── Load ML Model ───
MODEL = None
CLASS_LABELS = {}
DISEASE_INFO = {}

def load_ml_assets():
    global MODEL, CLASS_LABELS, DISEASE_INFO
    if TF_AVAILABLE and os.path.exists("plant_disease_model.tflite"):
        try:
            interpreter = tf.lite.Interpreter(model_path="plant_disease_model.tflite")
            interpreter.allocate_tensors()
            MODEL = interpreter
            print("✅  TFLite Model loaded successfully")
        except Exception as e:
            print(f"⚠️  Model load failed: {e}")

    if os.path.exists("class_indices.json"):
        with open("class_indices.json", "r") as f:
            idx = json.load(f)
            CLASS_LABELS = {v: k for k, v in idx.items()}

    if os.path.exists("disease_info.json"):
        with open("disease_info.json", "r", encoding="utf-8") as f:
            DISEASE_INFO = json.load(f)

load_ml_assets()

# ═══════════════════════════════════════════════
#  DATABASE
# ═══════════════════════════════════════════════
def get_db():
    conn = sqlite3.connect("database.db")
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db()
    c = conn.cursor()
    c.execute("""CREATE TABLE IF NOT EXISTS users (
        id       INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE NOT NULL,
        password TEXT NOT NULL,
        created  TEXT DEFAULT (datetime('now'))
    )""")
    c.execute("""CREATE TABLE IF NOT EXISTS predictions (
        id           INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id      INTEGER,
        filename     TEXT,
        disease_name TEXT,
        confidence   REAL,
        created      TEXT DEFAULT (datetime('now'))
    )""")
    c.execute("""CREATE TABLE IF NOT EXISTS feedbacks (
        id      INTEGER PRIMARY KEY AUTOINCREMENT,
        name    TEXT,
        text    TEXT,
        rating  INTEGER,
        created TEXT DEFAULT (datetime('now'))
    )""")
    conn.commit()
    conn.close()

init_db()

# ═══════════════════════════════════════════════
#  HELPERS
# ═══════════════════════════════════════════════
def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_image(img_path):
    img = keras_image.load_img(img_path, target_size=(224, 224))
    arr = keras_image.img_to_array(img)
    arr = np.expand_dims(arr, axis=0)
    arr = arr / 255.0
    return arr

def predict_disease(img_path):
    if MODEL is None:
        return None, 0.0
    arr = preprocess_image(img_path)
    input_details = MODEL.get_input_details()
    output_details = MODEL.get_output_details()
    MODEL.set_tensor(input_details[0]['index'], arr.astype(np.float32))
    MODEL.invoke()
    preds = MODEL.get_tensor(output_details[0]['index'])
    idx = int(np.argmax(preds, axis=1)[0])
    conf = round(float(np.max(preds)) * 100, 2)
    name = CLASS_LABELS.get(idx, "Unknown")
    return name, conf

def get_disease_info(name):
    return DISEASE_INFO.get(name, {
        "description": "No information available for this condition.",
        "treatment":   "Please consult an agricultural expert.",
        "pesticide":   "Not determined."
    })

# ═══════════════════════════════════════════════
#  ROUTES — PAGES
# ═══════════════════════════════════════════════
@app.route("/")
def home():
    return render_template("home.html")

@app.route("/upload", methods=["GET", "POST"])
def upload():
    if request.method == "POST":
        if "image" not in request.files:
            return jsonify({"error": "No file uploaded"}), 400

        file = request.files["image"]
        if file.filename == "" or not allowed_file(file.filename):
            return jsonify({"error": "Invalid file type"}), 400

        filename = secure_filename(file.filename)
        # avoid name clashes
        unique_name = f"{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}_{filename}"
        filepath = os.path.join(app.config["UPLOAD_FOLDER"], unique_name)
        file.save(filepath)

        disease_name, confidence = predict_disease(filepath)

        if disease_name is None:
            return jsonify({"error": "Model not loaded. Check server logs."}), 500

        info = get_disease_info(disease_name)

        # Save prediction to DB
        conn = get_db()
        conn.execute(
            "INSERT INTO predictions (user_id, filename, disease_name, confidence) VALUES (?,?,?,?)",
            (session.get("user_id"), unique_name, disease_name, confidence)
        )
        conn.commit()
        conn.close()

        is_healthy = "healthy" in disease_name.lower()

        return jsonify({
            "success":      True,
            "disease_name": disease_name,
            "confidence":   confidence,
            "is_healthy":   is_healthy,
            "description":  info["description"],
            "treatment":    info["treatment"],
            "pesticide":    info["pesticide"],
            "image_url":    f"/static/uploads/{unique_name}"
        })

    return render_template("upload.html")

@app.route("/history")
def history():
    if "user_id" not in session:
        return redirect(url_for("login"))
    conn = get_db()
    rows = conn.execute(
        "SELECT * FROM predictions WHERE user_id=? ORDER BY created DESC LIMIT 20",
        (session["user_id"],)
    ).fetchall()
    conn.close()
    return render_template("history.html", predictions=rows)

# ─── Auth ───
@app.route("/signup", methods=["GET", "POST"])
def signup():
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "").strip()
        if not username or not password:
            flash("Please fill in all fields.", "error")
            return render_template("signup.html")
        hashed = generate_password_hash(password)
        conn = get_db()
        try:
            conn.execute("INSERT INTO users (username, password) VALUES (?,?)", (username, hashed))
            conn.commit()
            flash("Account created! Please log in.", "success")
            return redirect(url_for("login"))
        except sqlite3.IntegrityError:
            flash("Username already taken.", "error")
        finally:
            conn.close()
    return render_template("signup.html")

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "").strip()
        conn = get_db()
        user = conn.execute("SELECT * FROM users WHERE username=?", (username,)).fetchone()
        conn.close()
        if user and check_password_hash(user["password"], password):
            session["user_id"] = user["id"]
            session["username"] = user["username"]
            return redirect(url_for("dashboard"))
        flash("Invalid username or password.", "error")
    return render_template("login.html")

@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))

@app.route("/dashboard")
def dashboard():
    if "user_id" not in session:
        return redirect(url_for("login"))
    conn = get_db()
    total = conn.execute("SELECT COUNT(*) FROM predictions WHERE user_id=?", (session["user_id"],)).fetchone()[0]
    recent = conn.execute(
        "SELECT * FROM predictions WHERE user_id=? ORDER BY created DESC LIMIT 5",
        (session["user_id"],)
    ).fetchall()
    conn.close()
    return render_template("dashboard.html", username=session["username"], total=total, recent=recent)

@app.route("/feedback", methods=["GET", "POST"])
def feedback():
    if request.method == "POST":
        name   = request.form.get("name", "").strip()
        text   = request.form.get("text", "").strip()
        rating = request.form.get("rating", 5)
        if name and text:
            conn = get_db()
            conn.execute("INSERT INTO feedbacks (name,text,rating) VALUES (?,?,?)", (name, text, rating))
            conn.commit()
            conn.close()
            return jsonify({"success": True})
        return jsonify({"error": "Please fill all fields"}), 400
    return render_template("feedback.html")

@app.route("/admin")
def admin():
    feedbacks   = []
    predictions = []
    conn = get_db()
    feedbacks   = conn.execute("SELECT * FROM feedbacks ORDER BY created DESC").fetchall()
    predictions = conn.execute("SELECT p.*, u.username FROM predictions p LEFT JOIN users u ON p.user_id=u.id ORDER BY p.created DESC LIMIT 50").fetchall()
    users_count = conn.execute("SELECT COUNT(*) FROM users").fetchone()[0]
    conn.close()
    return render_template("admin.html", feedbacks=feedbacks, predictions=predictions, users_count=users_count)

# ─── API endpoint for JS fetch ───
@app.route("/api/predict", methods=["POST"])
def api_predict():
    return upload()

# ═══════════════════════════════════════════════
#  CHATBOT
# ═══════════════════════════════════════════════
@app.route("/chatbot")
def chatbot():
    return render_template("chatbot.html")

@app.route("/api/chatbot", methods=["POST"])
def api_chatbot():
    """Proxy to Anthropic API for farming questions — with smart local fallback"""
    data = request.get_json(force=True)
    user_message = data.get("message", "").lower()
    lang = data.get("lang", "hi")
    system_prompt = data.get("system", "You are a helpful farming assistant for Indian farmers.")

    # Try Anthropic API first
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if api_key:
        try:
            import urllib.request
            payload = json.dumps({
                "model": "claude-haiku-4-5-20251001",
                "max_tokens": 500,
                "system": system_prompt,
                "messages": [{"role": "user", "content": data.get("message", "")}]
            }).encode("utf-8")
            req = urllib.request.Request(
                "https://api.anthropic.com/v1/messages",
                data=payload,
                headers={
                    "Content-Type": "application/json",
                    "x-api-key": api_key,
                    "anthropic-version": "2023-06-01"
                },
                method="POST"
            )
            with urllib.request.urlopen(req, timeout=15) as resp:
                result = json.loads(resp.read().decode("utf-8"))
                reply = result["content"][0]["text"]
                return jsonify({"response": reply})
        except Exception as e:
            print(f"Anthropic API error: {e}")

    # Smart local knowledge base fallback (no API key needed)
    hi = (lang == "hi")

    farming_responses = {
        # Diseases
        ("blast", "ब्लास्ट"): {
            "hi": "🌾 धान का ब्लास्ट रोग (Pyricularia oryzae):\n\n🔍 लक्षण: पत्तियों पर हीरे के आकार के भूरे धब्बे\n\n💊 उपचार:\n• Tricyclazole 75 WP — 0.6g/लीटर\n• Isoprothiolane 40 EC — 1.5ml/लीटर\n• Kasugamycin 3 SL — 2ml/लीटर\n\n📅 10-15 दिन के अंतर पर 2-3 बार छिड़काव करें\n⚠️ खेत में ज़्यादा पानी न रोकें",
            "en": "🌾 Rice Blast (Pyricularia oryzae):\n\n🔍 Symptoms: Diamond-shaped brown spots on leaves\n\n💊 Treatment:\n• Tricyclazole 75 WP — 0.6g/liter\n• Isoprothiolane 40 EC — 1.5ml/liter\n• Kasugamycin 3 SL — 2ml/liter\n\n📅 2-3 sprays at 10-15 day intervals\n⚠️ Avoid waterlogging"
        },
        ("blight", "झुलसा"): {
            "hi": "🍅 झुलसा रोग (Blight) उपचार:\n\n🍅 टमाटर/आलू Late Blight:\n• Metalaxyl + Mancozeb (Ridomil Gold) — 2g/लीटर\n• Mancozeb 75 WP — 2g/लीटर\n\n🌾 गेहूं Leaf Blight:\n• Propiconazole 25 EC — 1ml/लीटर\n\n📅 रोग दिखते ही तुरंत छिड़काव शुरू करें",
            "en": "🍅 Blight Disease Treatment:\n\n🍅 Tomato/Potato Late Blight:\n• Metalaxyl + Mancozeb — 2g/liter\n• Mancozeb 75 WP — 2g/liter\n\n🌾 Wheat Leaf Blight:\n• Propiconazole 25 EC — 1ml/liter\n\n📅 Start spraying immediately when symptoms appear"
        },
        ("rust", "जंग", "रतुआ"): {
            "hi": "🌾 जंग रोग (Rust) उपचार:\n\n💊 कीटनाशक:\n• Propiconazole 25 EC — 1ml/लीटर\n• Tebuconazole 25.9 EC — 1ml/लीटर\n• Hexaconazole 5 SC — 2ml/लीटर\n\n📅 14 दिन के अंतर पर 2 बार छिड़काव\n✅ रोग प्रतिरोधक किस्में चुनें",
            "en": "🌾 Rust Disease Treatment:\n\n💊 Fungicides:\n• Propiconazole 25 EC — 1ml/liter\n• Tebuconazole 25.9 EC — 1ml/liter\n• Hexaconazole 5 SC — 2ml/liter\n\n📅 2 sprays at 14-day intervals\n✅ Choose resistant varieties"
        },
        ("mildew", "पाउडरी", "powdery"): {
            "hi": "🍃 पाउडरी मिल्ड्यू उपचार:\n\n💊 फफूंदनाशक:\n• Sulphur 80 WG — 3g/लीटर (सबसे सस्ता)\n• Carbendazim 50 WP — 1g/लीटर\n• Hexaconazole 5 SC — 2ml/लीटर\n\n🌿 जैविक: Neem oil 5ml/लीटर + Baking soda 5g/लीटर\n📅 7-10 दिन में एक बार छिड़काव",
            "en": "🍃 Powdery Mildew Treatment:\n\n💊 Fungicides:\n• Sulphur 80 WG — 3g/liter (cheapest)\n• Carbendazim 50 WP — 1g/liter\n• Hexaconazole 5 SC — 2ml/liter\n\n🌿 Organic: Neem oil 5ml/L + Baking soda 5g/L\n📅 Spray every 7-10 days"
        },
        ("aphid", "एफिड", "whitefly", "सफेद मक्खी", "thrips", "थ्रिप्स"): {
            "hi": "🐛 चूसने वाले कीट नियंत्रण:\n\n💊 कीटनाशक:\n• Imidacloprid 17.8 SL — 0.5ml/लीटर\n• Thiamethoxam 25 WG — 0.5g/लीटर\n• Neem oil 5ml/लीटर (जैविक)\n\n⚠️ एक ही कीटनाशक बार-बार न उपयोग करें\n📅 7 दिन के अंतर पर 2-3 बार",
            "en": "🐛 Sucking Pest Control:\n\n💊 Insecticides:\n• Imidacloprid 17.8 SL — 0.5ml/liter\n• Thiamethoxam 25 WG — 0.5g/liter\n• Neem oil 5ml/liter (organic)\n\n⚠️ Rotate insecticides to avoid resistance\n📅 2-3 sprays at 7-day intervals"
        },
        ("wheat", "गेहूं"): {
            "hi": "🌾 गेहूं की खेती:\n\n📅 बुआई: नवंबर-दिसंबर (उत्तर भारत)\n🌱 बीज दर: 100-125 किग्रा/हेक्टेयर\n💧 सिंचाई: 4-6 बार (CRI, Tillering, Jointing, Booting, Milking)\n\n✅ उन्नत किस्में:\n• HD-2967, DBW-17 (उत्तर भारत)\n• K-307, PBW-343 (पंजाब/हरियाणा)\n\n🌡️ उपयुक्त तापमान: 15-25°C\n💊 मुख्य रोग: जंग, करनाल बंट — Propiconazole से उपचार",
            "en": "🌾 Wheat Cultivation:\n\n📅 Sowing: November-December (North India)\n🌱 Seed rate: 100-125 kg/hectare\n💧 Irrigation: 4-6 times (CRI, Tillering, Jointing, Booting, Milking)\n\n✅ Good varieties:\n• HD-2967, DBW-17 (North India)\n• K-307, PBW-343 (Punjab/Haryana)\n\n🌡️ Ideal temp: 15-25°C\n💊 Main diseases: Rust, Karnal Bunt — treat with Propiconazole"
        },
        ("paddy", "rice", "धान", "चावल"): {
            "hi": "🌾 धान की खेती:\n\n📅 नर्सरी: मई-जून | रोपाई: जून-जुलाई\n🌱 बीज दर: 20-25 किग्रा/हेक्टेयर\n💧 पानी: 5-7 सेमी खड़ा रखें\n\n✅ किस्में:\n• पूसा 1121 (बासमती)\n• MTU-7029, Samba Mahsuri\n• HKR-47 (हरियाणा)\n\n💊 मुख्य रोग:\n• ब्लास्ट: Tricyclazole 75 WP\n• भूरा धब्बा: Mancozeb 75 WP\n• बकानी: Carbendazim 50 WP",
            "en": "🌾 Paddy Cultivation:\n\n📅 Nursery: May-June | Transplanting: June-July\n🌱 Seed rate: 20-25 kg/hectare\n💧 Maintain 5-7 cm standing water\n\n✅ Varieties:\n• Pusa 1121 (Basmati)\n• MTU-7029, Samba Mahsuri\n\n💊 Main diseases:\n• Blast: Tricyclazole 75 WP\n• Brown spot: Mancozeb 75 WP\n• Bakanae: Carbendazim 50 WP"
        },
        ("tomato", "टमाटर"): {
            "hi": "🍅 टमाटर की खेती:\n\n📅 रोपाई: जून-जुलाई और अक्टूबर-नवंबर\n📏 दूरी: 60×45 सेमी\n💧 सिंचाई: 7-10 दिन में\n\n✅ किस्में: Pusa Ruby, Solan Vajra, Hybrid\n\n💊 रोग प्रबंधन:\n• Early Blight: Mancozeb 2g/L\n• Late Blight: Metalaxyl+Mancozeb 2g/L\n• Leaf Curl Virus: Imidacloprid 0.5ml/L\n\n🌱 खाद: DAP 200kg + MOP 100kg + Urea 200kg/हेक्टेयर",
            "en": "🍅 Tomato Cultivation:\n\n📅 Transplanting: June-July and October-November\n📏 Spacing: 60×45 cm\n💧 Irrigation: Every 7-10 days\n\n✅ Varieties: Pusa Ruby, Solan Vajra, Hybrids\n\n💊 Disease management:\n• Early Blight: Mancozeb 2g/L\n• Late Blight: Metalaxyl+Mancozeb 2g/L\n• Leaf Curl: Imidacloprid 0.5ml/L"
        },
        ("potato", "आलू"): {
            "hi": "🥔 आलू की खेती:\n\n📅 बुआई: अक्टूबर-नवंबर\n🌱 बीज दर: 25-30 क्विंटल/हेक्टेयर\n📏 दूरी: 60×25 सेमी\n💧 सिंचाई: 8-10 दिन में\n\n✅ किस्में: Kufri Jyoti, Kufri Bahar, Kufri Sindhuri\n\n💊 मुख्य रोग:\n• Late Blight: Metalaxyl+Mancozeb 2g/L\n• Early Blight: Mancozeb 2g/L\n• Scab: Sulphur 80WG से बीज उपचार\n\n🌡️ 15-20°C तापमान सर्वोत्तम",
            "en": "🥔 Potato Cultivation:\n\n📅 Sowing: October-November\n🌱 Seed rate: 25-30 quintal/hectare\n💧 Irrigation: Every 8-10 days\n\n✅ Varieties: Kufri Jyoti, Kufri Bahar, Kufri Sindhuri\n\n💊 Main diseases:\n• Late Blight: Metalaxyl+Mancozeb 2g/L\n• Early Blight: Mancozeb 2g/L\n\n🌡️ Ideal temperature: 15-20°C"
        },
        ("soil", "मिट्टी", "ph", "testing"): {
            "hi": "🌍 मिट्टी जांच और प्रबंधन:\n\n🔬 मुफ्त जांच कहाँ:\n• कृषि विज्ञान केंद्र (KVK)\n• जिला कृषि प्रयोगशाला\n• मृदा स्वास्थ्य कार्ड योजना\n\n📊 pH प्रबंधन:\n• pH 6-7.5 — अधिकांश फसलों के लिए ठीक\n• pH कम (<6): 100-200 kg/हेक्टेयर चूना\n• pH अधिक (>7.5): 10-20 kg/हेक्टेयर सल्फर\n\n🌱 जैव खाद: Vermicompost 3-4 टन/हेक्टेयर",
            "en": "🌍 Soil Testing & Management:\n\n🔬 Free testing at:\n• Krishi Vigyan Kendra (KVK)\n• District Agriculture Lab\n• Soil Health Card Scheme\n\n📊 pH Management:\n• pH 6-7.5 — ideal for most crops\n• Low pH (<6): Apply lime 100-200 kg/ha\n• High pH (>7.5): Apply sulphur 10-20 kg/ha\n\n🌱 Organic: Vermicompost 3-4 ton/ha"
        },
        ("fertilizer", "खाद", "urea", "dap", "npk"): {
            "hi": "🌱 खाद (उर्वरक) प्रबंधन:\n\n📊 मुख्य तत्व:\n• N (यूरिया 46%): पत्तियों की बढ़वार\n• P (DAP): जड़ और फूल विकास\n• K (MOP 60%): रोग प्रतिरोधक, फल गुणवत्ता\n\n📅 कब डालें:\n• बुआई: DAP 100kg + MOP 50kg/हेक्टेयर\n• 30 दिन: यूरिया 65kg\n• 60 दिन: यूरिया 65kg\n\n💡 सूक्ष्म तत्व: Zinc Sulphate 25kg/हेक्टेयर\n⚠️ मिट्टी जांच के बाद ही खाद डालें",
            "en": "🌱 Fertilizer Management:\n\n📊 Key nutrients:\n• N (Urea 46%): leaf growth\n• P (DAP): root & flower development\n• K (MOP 60%): disease resistance, fruit quality\n\n📅 Application timing:\n• At sowing: DAP 100kg + MOP 50kg/ha\n• Day 30: Urea 65kg/ha\n• Day 60: Urea 65kg/ha\n\n💡 Micronutrients: Zinc Sulphate 25kg/ha\n⚠️ Always test soil before applying fertilizers"
        },
        ("irrigation", "सिंचाई", "water", "पानी"): {
            "hi": "💧 सिंचाई प्रबंधन:\n\n🔧 विधियाँ:\n• ड्रिप: 40-50% पानी बचत — सब्जियों के लिए\n• फव्वारा: गेहूं, चना, सब्जियों के लिए\n• नाली: धान, गन्ना के लिए\n\n📅 फसल-वार सिंचाई:\n• गेहूं: बुआई + 4-5 बार (CRI, Tillering, Jointing, Booting, Milking)\n• धान: हर समय 5-7 सेमी पानी\n• टमाटर/सब्जी: 7-10 दिन में\n\n💡 सुबह 6-10 बजे सिंचाई सबसे अच्छी",
            "en": "💧 Irrigation Management:\n\n🔧 Methods:\n• Drip: 40-50% water saving — best for vegetables\n• Sprinkler: For wheat, chickpea, vegetables\n• Flood/furrow: For paddy, sugarcane\n\n📅 Crop-wise schedule:\n• Wheat: 4-5 times (CRI, Tillering, Jointing, Booting, Milking)\n• Paddy: Maintain 5-7 cm water always\n• Tomato/vegetables: Every 7-10 days\n\n💡 Morning irrigation (6-10 AM) is best"
        },
        ("government", "scheme", "सरकारी", "योजना", "pm-kisan", "pmkisan", "subsidy"): {
            "hi": "🏛️ किसान कल्याण योजनाएं:\n\n💰 PM-KISAN: ₹6,000/साल — pmkisan.gov.in\n🛡️ PMFBY (फसल बीमा): 2% प्रीमियम पर बीमा — pmfby.gov.in\n💳 KCC (किसान क्रेडिट कार्ड): 4% ब्याज पर ऋण\n🌍 मृदा स्वास्थ्य कार्ड: मुफ्त मिट्टी जांच\n🌾 eNAM: ऑनलाइन मंडी — enam.gov.in\n🌱 PKVY: जैविक खेती सहायता\n\n📞 Kisan Call Centre: 1800-180-1551 (निःशुल्क)",
            "en": "🏛️ Farmer Welfare Schemes:\n\n💰 PM-KISAN: ₹6,000/year — pmkisan.gov.in\n🛡️ PMFBY (Crop Insurance): Only 2% premium — pmfby.gov.in\n💳 KCC (Kisan Credit Card): Loan at 4% interest\n🌍 Soil Health Card: Free soil testing\n🌾 eNAM: Online market — enam.gov.in\n🌱 PKVY: Organic farming support\n\n📞 Kisan Call Centre: 1800-180-1551 (free)"
        },
        ("organic", "जैविक", "bio"): {
            "hi": "🌿 जैविक खेती:\n\n✅ जैविक खाद:\n• Vermicompost: 3-4 टन/हेक्टेयर\n• FYM (गोबर खाद): 10-15 टन/हेक्टेयर\n• Jeevamrit: 200L पानी + 10kg गोबर + 10L गोमूत्र\n\n🐛 जैविक कीट नियंत्रण:\n• Neem oil 5ml/लीटर\n• Trichoderma: मिट्टी में मिलाएं\n• Beauveria bassiana: सफेद मक्खी के लिए\n\n📜 सरकारी सहायता: PKVY योजना में ₹50,000/हेक्टेयर",
            "en": "🌿 Organic Farming:\n\n✅ Organic manures:\n• Vermicompost: 3-4 ton/ha\n• FYM: 10-15 ton/ha\n• Jeevamrit: 200L water + 10kg cowdung + 10L cow urine\n\n🐛 Biopesticides:\n• Neem oil 5ml/liter\n• Trichoderma: Mix in soil\n• Beauveria bassiana: For whitefly\n\n📜 Govt support: PKVY scheme ₹50,000/ha"
        },
    }

    # Match question to response
    for keys, response_dict in farming_responses.items():
        if any(k in user_message for k in (keys if isinstance(keys, tuple) else (keys,))):
            reply = response_dict.get("hi" if hi else "en", response_dict.get("en", response_dict.get("hi", "")))
            return jsonify({"response": reply})

    # Generic helpful fallback
    if hi:
        fb = ("मुझे खेद है, लेकिन मैं आपके इस प्रश्न का सीधा जवाब नहीं दे पाया। "
              "कृपया इनमें से कोई विषय पूछें:\n\n"
              "🌾 फसल: गेहूं, धान, टमाटर, आलू, मक्का, सरसों\n"
              "🐛 रोग: ब्लास्ट, झुलसा, जंग, पाउडरी मिल्ड्यू, एफिड\n"
              "💧 सिंचाई प्रबंधन\n🌱 खाद (NPK, DAP, यूरिया)\n"
              "🏛️ सरकारी योजनाएं\n🌿 जैविक खेती\n\n"
              "📞 या Kisan Call Centre: 1800-180-1551 पर call करें (Free)")
    else:
        fb = ("I couldn't find a specific answer for your question. "
              "Please ask about:\n\n"
              "🌾 Crops: wheat, paddy, tomato, potato, maize, mustard\n"
              "🐛 Diseases: blast, blight, rust, powdery mildew, aphids\n"
              "💧 Irrigation management\n🌱 Fertilizers (NPK, DAP, Urea)\n"
              "🏛️ Government schemes\n🌿 Organic farming\n\n"
              "📞 Or call Kisan Call Centre: 1800-180-1551 (Free)")
    return jsonify({"response": fb})


# ═══════════════════════════════════════════════
#  ENHANCED FEEDBACK
# ═══════════════════════════════════════════════

# Extend feedbacks table with category & audio support
def upgrade_db():
    conn = get_db()
    c = conn.cursor()
    # Add category column if not exists
    try:
        c.execute("ALTER TABLE feedbacks ADD COLUMN category TEXT DEFAULT 'General'")
    except Exception:
        pass
    try:
        c.execute("ALTER TABLE feedbacks ADD COLUMN has_audio INTEGER DEFAULT 0")
    except Exception:
        pass
    try:
        c.execute("ALTER TABLE feedbacks ADD COLUMN audio_path TEXT")
    except Exception:
        pass
    # Orders table
    c.execute("""CREATE TABLE IF NOT EXISTS orders (
        id         INTEGER PRIMARY KEY AUTOINCREMENT,
        order_id   TEXT UNIQUE,
        name       TEXT,
        phone      TEXT,
        address    TEXT,
        pincode    TEXT,
        payment    TEXT,
        items_json TEXT,
        total      REAL,
        status     TEXT DEFAULT 'Confirmed',
        created    TEXT DEFAULT (datetime('now'))
    )""")
    conn.commit()
    conn.close()

upgrade_db()

AUDIO_FOLDER = os.path.join("static", "audio_feedback")
os.makedirs(AUDIO_FOLDER, exist_ok=True)

@app.route("/feedback/voice", methods=["POST"])
def feedback_voice():
    name   = request.form.get("name", "").strip()
    text   = request.form.get("text", "[Voice feedback]").strip()
    rating = request.form.get("rating", 5)
    category = request.form.get("category", "Voice Feedback")

    if not name:
        return jsonify({"error": "Name required"}), 400

    audio_path = None
    has_audio = 0
    if "audio" in request.files:
        audio_file = request.files["audio"]
        if audio_file.filename:
            fname = f"voice_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}_{session.get('user_id','anon')}.webm"
            audio_path = os.path.join(AUDIO_FOLDER, fname)
            audio_file.save(audio_path)
            has_audio = 1

    conn = get_db()
    conn.execute(
        "INSERT INTO feedbacks (name, text, rating, category, has_audio, audio_path) VALUES (?,?,?,?,?,?)",
        (name, text, rating, category, has_audio, audio_path)
    )
    conn.commit()
    conn.close()
    return jsonify({"success": True})

@app.route("/api/feedbacks")
def api_feedbacks():
    conn = get_db()
    rows = conn.execute(
        "SELECT id, name, text, rating, category, has_audio, created FROM feedbacks ORDER BY created DESC LIMIT 20"
    ).fetchall()
    conn.close()
    return jsonify({"feedbacks": [dict(r) for r in rows]})


# ═══════════════════════════════════════════════
#  SHOP / E-COMMERCE
# ═══════════════════════════════════════════════
@app.route("/shop")
def shop():
    return render_template("shop.html")

@app.route("/api/order", methods=["POST"])
def api_order():
    data = request.get_json(force=True)
    try:
        conn = get_db()
        conn.execute(
            "INSERT INTO orders (order_id, name, phone, address, pincode, payment, items_json, total) VALUES (?,?,?,?,?,?,?,?)",
            (
                data.get("order_id"),
                data.get("name"),
                data.get("phone"),
                data.get("address"),
                data.get("pincode"),
                data.get("payment"),
                json.dumps(data.get("items", [])),
                data.get("total", 0)
            )
        )
        conn.commit()
        conn.close()
        return jsonify({"success": True, "order_id": data.get("order_id")})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
