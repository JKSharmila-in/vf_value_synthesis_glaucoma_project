import os
from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename

from inference import predict_vf


# -----------------------------------
# BASIC APP CONFIG
# -----------------------------------
app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16 MB


# -----------------------------------
# ENSURE UPLOAD FOLDER EXISTS
# -----------------------------------
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


# -----------------------------------
# ROUTES (HTML PAGES)
# -----------------------------------
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/patient-details")
def patient_details():
    return render_template("patient-details.html")


@app.route("/prediction-result")
def prediction_result():
    return render_template("prediction-result.html")


@app.route("/feedback")
def feedback():
    return render_template("feedback.html")


# -----------------------------------
# API ROUTE (MODEL INFERENCE)
# -----------------------------------
@app.route("/predict", methods=["POST"])
def predict():
    if "fundus_image" not in request.files:
        return jsonify({"error": "No fundus image uploaded"}), 400

    image_file = request.files["fundus_image"]

    if image_file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    filename = secure_filename(image_file.filename)
    image_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    image_file.save(image_path)

    try:
        vf_values = predict_vf(image_path)

        return jsonify({
            "success": True,
            "vf_values": vf_values
        })

    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


# -----------------------------------
# MAIN
# -----------------------------------
if __name__ == "__main__":
    app.run(debug=True)
