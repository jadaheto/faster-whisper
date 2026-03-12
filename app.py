import os
import uuid
import tempfile
from flask import Flask, request, jsonify
from faster_whisper import WhisperModel

app = Flask(__name__)

MODEL_SIZE = os.getenv("MODEL_SIZE", "small")
DEVICE = os.getenv("DEVICE", "cpu")
COMPUTE_TYPE = os.getenv("COMPUTE_TYPE", "int8")
MAX_CONTENT_LENGTH = int(os.getenv("MAX_CONTENT_LENGTH", 25 * 1024 * 1024))

app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH

model = WhisperModel(MODEL_SIZE, device=DEVICE, compute_type=COMPUTE_TYPE)


@app.route("/", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


@app.route("/transcribe", methods=["POST"])
def transcribe():
    if "audio" not in request.files:
        return jsonify({"error": "Falta el archivo audio"}), 400

    audio = request.files["audio"]
    filename = audio.filename.lower()

    allowed_extensions = {".mp3", ".wav", ".m4a", ".ogg", ".flac", ".webm"}
    if not any(filename.endswith(ext) for ext in allowed_extensions):
        return jsonify({"error": "Tipo de archivo no permitido"}), 400

    temp_path = None
    try:
        suffix = os.path.splitext(filename)[1] or ".tmp"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            temp_path = tmp.name
            audio.save(temp_path)

        segments, info = model.transcribe(temp_path, beam_size=5)
        result = [{"start": s.start, "end": s.end, "text": s.text} for s in segments]

        return jsonify({"language": info.language, "segments": result})
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)


if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
