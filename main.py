import os
import re
import string
import html
import joblib
import telebot
import numpy as np

from nltk.tokenize import TweetTokenizer


# ==========================================
# 1. CONFIGURATION
# ==========================================

BOT_TOKEN = os.environ.get("BOT_TOKEN")

if not BOT_TOKEN:
    raise ValueError("Telegram bot token is missing!")

bot = telebot.TeleBot(BOT_TOKEN)


# ==========================================
# 2. LOAD TRAINED MODEL
# ==========================================

model = joblib.load("hate_speech_model.pkl")

print("Hate Speech Detection Model Loaded!")


# ==========================================
# 3. TEXT PREPROCESSING
# ==========================================

def preprocess_text(text):
    text = str(text)
    text = text.lower()

    text = re.sub(
        r"https?://\S+|www\.\S+",
        " ",
        text
    )

    text = re.sub(
        r"\b[\w.+-]+@[\w.-]+\.[a-zA-Z]{2,}\b",
        " ",
        text
    )

    text = re.sub(r"(?<!\w)@\w+", " ", text)

    text = text.translate(
        str.maketrans("", "", string.punctuation)
    )

    text = re.sub(r"\s+", " ", text).strip()

    return text


# ==========================================
# 4. CLASSIFICATION FUNCTION
# ==========================================

def predict_hate_speech(text):

    cleaned_text = preprocess_text(text)

    if not cleaned_text:
        return None

    prediction = model.predict([cleaned_text])[0]

    probabilities = model.predict_proba([cleaned_text])[0]

    best_index = np.argmax(probabilities)

    confidence = probabilities[best_index] * 100

    results = {
        "hate_speech": "🔴 Hate Speech",
        "offensive": "🟠 Offensive Language",
        "normal": "🟢 Normal"
    }

    return {
        "prediction": prediction,
        "result": results.get(prediction, prediction),
        "confidence": confidence,
        "probabilities": dict(
            zip(model.classes_, probabilities * 100)
        )
    }


# ==========================================
# 5. START COMMAND
# ==========================================

@bot.message_handler(commands=["start"])
def start(message):

    welcome_message = (
        "👋 <b>Welcome to Hate Speech Detection Bot!</b>\n\n"
        "I can analyze text and classify it into three categories:\n\n"
        "🔴 Hate Speech\n"
        "🟠 Offensive Language\n"
        "🟢 Normal\n\n"
        "Send me any text message to get started."
    )

    bot.reply_to(
        message,
        welcome_message,
        parse_mode="HTML"
    )


# ==========================================
# 6. HELP COMMAND
# ==========================================

@bot.message_handler(commands=["help"])
def help_command(message):

    bot.reply_to(
        message,
        "Send me a text message, and I will analyze it "
        "using a Machine Learning model."
    )


# ==========================================
# 7. HANDLE TEXT MESSAGES
# ==========================================

@bot.message_handler(
    content_types=["text"],
    func=lambda message: not message.text.startswith("/")
)
def handle_message(message):

    user_text = message.text

    try:

        result = predict_hate_speech(user_text)

        if result is None:
            bot.reply_to(message, "Please enter valid text.")
            return

        prediction = html.escape(result["result"])

        response = (
            "🔍 <b>HATE SPEECH DETECTION RESULT</b>\n"
            "━━━━━━━━━━━━━━━━━━━━\n\n"
            f"📝 <b>Input:</b> {html.escape(user_text)}\n\n"
            f"📊 <b>Prediction:</b> {prediction}\n"
            f"🎯 <b>Confidence:</b> {result['confidence']:.2f}%\n\n"
            "<b>Class Probabilities:</b>\n"
        )

        for label, probability in result["probabilities"].items():

            response += (
                f"• {html.escape(label)}: {probability:.2f}%\n"
            )

        if result["confidence"] < 50:
            response += (
                "\n⚠️ <i>Low confidence: "
                "The prediction may be unreliable.</i>"
            )

        response += (
            "\n\n<i>Automated prediction. "
            "Results may not always be accurate.</i>"
        )

        bot.reply_to(
            message,
            response,
            parse_mode="HTML"
        )

    except Exception as error:

        print("Prediction error:", error)

        bot.reply_to(
            message,
            "Sorry, something went wrong while analyzing your message."
        )


# ==========================================
# 8. RUN TELEGRAM BOT
# ==========================================


bot.infinity_polling(
     skip_pending=True,
    timeout=30,
    long_polling_timeout=30
)