# 🌿 LeafScan AI — Plant Disease Detection

A full-stack Flask web app that uses your trained VGG-16 model to detect plant diseases from leaf photos in real-time.

---

## 📁 Project Structure

```
leafscan_project/
│
├── app.py                    ← Flask backend (this file)
├── plant_disease_model.h5    ← Your trained Keras model (copy here)
├── class_indices.json        ← Class label mapping (copy here)
├── disease_info.json         ← Disease descriptions/treatments (copy here)
├── database.db               ← Auto-created SQLite database
├── requirements.txt          ← Python dependencies
│
├── templates/
│   ├── base.html             ← Shared nav/footer layout
│   ├── home.html             ← Landing page
│   ├── upload.html           ← Main detector page ⭐
│   ├── login.html
│   ├── signup.html
│   ├── dashboard.html
│   ├── history.html
│   ├── feedback.html
│   └── admin.html
│
└── static/
    └── uploads/              ← Uploaded images stored here (auto-created)
```

---

## ⚙️ Setup Instructions

### 1. Copy your model files into the project root:
```
plant_disease_model.h5
class_indices.json
disease_info.json
```

### 2. Install dependencies:
```bash
pip install -r requirements.txt
```

### 3. Run the app:
```bash
python app.py
```

### 4. Open in browser:
```
http://localhost:5000
```

---

## 🔑 Key Routes

| Route | Description |
|-------|-------------|
| `/` | Home / landing page |
| `/upload` | **Main disease detector** — upload photo here |
| `/login` | User login |
| `/signup` | Create account |
| `/dashboard` | User's personal dashboard |
| `/history` | Scan history (logged-in users) |
| `/feedback` | Submit feedback |
| `/admin` | Admin panel (see all scans + feedbacks) |

---

## 🧠 How It Works

1. User uploads a leaf image on `/upload`
2. Flask saves it to `static/uploads/`
3. The image is preprocessed to 224×224 (VGG-16 input size)
4. `model.predict()` returns confidence scores for all 4 classes
5. The top class + confidence is returned as JSON
6. Frontend JavaScript renders the result card dynamically
7. If user is logged in, the prediction is saved to `database.db`

---

## 📝 Adding More Diseases

To expand beyond 4 Apple diseases:
1. Retrain your model with more classes on Google Colab
2. Update `class_indices.json` with new class names
3. Add entries to `disease_info.json` for descriptions/treatments
4. Replace `plant_disease_model.h5` with the new model

---

## 🔒 Security Notes

- Passwords are hashed with `werkzeug.security` (bcrypt)
- File uploads are sanitized with `secure_filename`
- Only image files (jpg, png, webp, bmp) are accepted
- Secret key should be changed for production

---

## 🚀 Deploy to Production

For production deployment:
```bash
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:5000 app:app
```

Or use Render, Railway, or any Python hosting platform.
