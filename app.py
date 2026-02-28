from flask import Flask, request
import requests
import os
from PIL import Image, ImageOps
import numpy as np
import tflite_runtime.interpreter as tflite

app = Flask(__name__)

# ====== REPLACE THESE ======
VERIFY_TOKEN = "your_verify_token"
ACCESS_TOKEN = "EAARZAVGHPLpQBQzx6OdybIic1crCf0m9dr24sa0rHx16ZAJHRzaekCZBuDvZANM4JUWhwgj1mVAU4SMZCUZCedIlihHgaFNSa8GXJzyP1j4ZCynKTFW35ZAvnccVTEsaGdSxN5hX53DUoZBph3zLZAVEpm8IKZBd4F8B8R3uiZCorpKbtlnIH9RHeqa6wVURfOpILusc6gZDZD"
PHONE_NUMBER_ID = "1005973195933095"
# ===========================


# ================= LOAD MODEL ONCE =================
interpreter = tflite.Interpreter(model_path="model.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

with open("labels.txt", "r") as f:
    class_names = f.readlines()
# ====================================================


DISEASE_MANAGEMENT = {
    "Bacterial Leaf Blight": "⚠️ *Bacterial Leaf Blight* detected.\n\n✅ Streptomycin sulphate + Copper oxychloride spray recommended.",
    "Brown Spot": "⚠️ *Brown Spot* detected.\n\n✅ Mancozeb 2g/litre spray recommended.",
    "Healthy Rice Leaf": "🌿 Good news! The paddy leaf is healthy.",
    "Leaf Blast": "⚠️ *Leaf Blast* detected.\n\n✅ Tricyclazole 75WP spray recommended.",
    "Leaf scald": "⚠️ *Leaf Scald* detected.\n\n✅ Hexaconazole spray recommended.",
    "Sheath Blight": "⚠️ *Sheath Blight* detected.\n\n✅ Carbendazim spray recommended."
}


def predict_paddy_disease(image_path):
    image = Image.open(image_path).convert("RGB")
    image = ImageOps.fit(image, (224, 224), Image.Resampling.LANCZOS)

    image_array = np.asarray(image).astype(np.float32)
    normalized_image_array = (image_array / 127.5) - 1
    input_data = np.expand_dims(normalized_image_array, axis=0)

    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()

    output_data = interpreter.get_tensor(output_details[0]['index'])
    index = np.argmax(output_data)

    class_name = class_names[index].strip()
    if " " in class_name:
        class_name = class_name.split(" ", 1)[1]

    confidence_score = float(output_data[0][index])

    return class_name, confidence_score


@app.route("/webhook", methods=["GET"])
def verify():
    mode = request.args.get("hub.mode")
    token = request.args.get("hub.verify_token")
    challenge = request.args.get("hub.challenge")

    if mode and token:
        if mode == "subscribe" and token == VERIFY_TOKEN:
            return challenge, 200

    return "Verification failed", 403


@app.route("/webhook", methods=["POST"])
def webhook():
    data = request.json

    try:
        message = data["entry"][0]["changes"][0]["value"]["messages"][0]
        sender = message["from"]

        if message["type"] == "image":
            media_id = message["image"]["id"]
            media_url = get_media_url(media_id)

            download_image(media_url, media_id)
            local_image_path = f"images/{media_id}.jpg"

            send_whatsapp_message(sender, "🔍 Image received. Analyzing...")

            try:
                predicted_class, confidence = predict_paddy_disease(local_image_path)
                advice = DISEASE_MANAGEMENT.get(
                    predicted_class,
                    "Diagnosis complete. Please consult agricultural expert."
                )

                reply_text = (
                    f"📊 *Diagnosis:* {predicted_class}\n"
                    f"🎯 *Confidence:* {confidence * 100:.1f}%\n\n"
                    f"{advice}"
                )

            except Exception as e:
                print("Prediction error:", e)
                reply_text = "Error during prediction. Please send a clear image."

        else:
            reply_text = "🌱 Please send a paddy leaf image for disease detection."

        send_whatsapp_message(sender, reply_text)

    except Exception as e:
        print("Webhook error:", e)

    return "OK", 200


def get_media_url(media_id):
    url = f"https://graph.facebook.com/v19.0/{media_id}"
    headers = {"Authorization": f"Bearer {ACCESS_TOKEN}"}
    response = requests.get(url, headers=headers)
    return response.json()["url"]


def download_image(url, media_id):
    headers = {"Authorization": f"Bearer {ACCESS_TOKEN}"}
    response = requests.get(url, headers=headers)

    if not os.path.exists("images"):
        os.makedirs("images")

    with open(f"images/{media_id}.jpg", "wb") as f:
        f.write(response.content)


def send_whatsapp_message(to, text):
    url = f"https://graph.facebook.com/v19.0/{PHONE_NUMBER_ID}/messages"

    headers = {
        "Authorization": f"Bearer {ACCESS_TOKEN}",
        "Content-Type": "application/json"
    }

    payload = {
        "messaging_product": "whatsapp",
        "to": to,
        "type": "text",
        "text": {"body": text}
    }

    requests.post(url, headers=headers, json=payload)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
