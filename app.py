from flask import Flask, request, render_template, jsonify, send_file
import pandas as pd
import joblib
import os

app = Flask(__name__)

# Load model
model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")
mask = joblib.load("mask.pkl")

@app.route("/", methods=["GET", "POST"])
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    text = [data["text"]]

    X = vectorizer.transform(text)
    X_selected = X[:, mask]
    pred = model.predict(X_selected)[0]
    sentiment = "Positif" if pred == 1 else "Negatif"

    return jsonify({"sentiment": sentiment})

@app.route("/analyze_csv", methods=["POST"])
def analyze_csv():
    if "file" not in request.files:
        return jsonify({"error": "Tidak ada file"}), 400
    
    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "Nama file kosong"}), 400

    # Baca CSV
    df = pd.read_csv(file)

    if "Tweet" not in df.columns:
        return jsonify({"error": "CSV harus punya kolom 'Tweet'"}), 400

    # Prediksi untuk setiap baris
    X = vectorizer.transform(df["Tweet"].astype(str))
    X_selected = X[:, mask]
    preds = model.predict(X_selected)

    df["Sentiment"] = ["Positif" if p == 1 else "Negatif" for p in preds]

    # Simpan hasil ke file sementara
    output_path = "hasil_prediksi.csv"
    df.to_csv(output_path, index=False, encoding="utf-8")

    # Return hasil sebagai tabel JSON
    return jsonify({
        "columns": df.columns.tolist(),
        "data": df.head(20).values.tolist(),  # tampilkan 20 baris pertama
        "download_link": "/download"
    })

@app.route("/download")
def download_file():
    return send_file("hasil_prediksi.csv", as_attachment=True)

if __name__ == "__main__":
    app.run(debug=True)
