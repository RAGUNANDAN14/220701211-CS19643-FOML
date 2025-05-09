import random
import pickle

# Load model and vectorizer
with open("chatbot/model.pkl", "rb") as f:
    model = pickle.load(f)
with open("chatbot/vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

# Dictionary of emotion-based responses
mood_keywords = {
    "anxious": [
        "You're feeling anxious. Try this: Inhale for 4s, hold for 7s, exhale for 8s. 🧘",
        "You're not alone. Let's take a few deep breaths together. 🌿",
        "Try box breathing: inhale 4s, hold 4s, exhale 4s, hold 4s. 🫁"
    ],
    "sad": [
        "Sadness is part of healing. I'm here with you. 💙",
        "Take a moment to rest and be kind to yourself. 🌧️",
        "It's okay to feel down. Let's breathe slowly together. 🌱"
    ],
    "happy": [
        "That's wonderful! Keep smiling! 🌞",
        "Joy is contagious—spread it around! 🎉",
        "What made you happy today? 😊"
    ],
    # Add ~100 total categories similarly
    "stressed": [
        "Feeling stressed? Try a 5-minute meditation. 🧘",
        "Write down your thoughts and let them go. 📝",
        "Try a breathing exercise or go for a short walk. 🚶"
    ],
    "tired": [
        "Take a short break or nap if you can. 💤",
        "Recharge by stretching or stepping outside. 🌤️",
        "Tired minds need care. Rest is productive too. 🛏️"
    ]
}

def get_bot_response(user_input):
    user_input_lower = user_input.lower()

    # Predict emotion
    X_input = vectorizer.transform([user_input_lower])
    predicted_emotion = model.predict(X_input)[0]

    if predicted_emotion in mood_keywords:
        return random.choice(mood_keywords[predicted_emotion])

    return "I'm here for you. Tell me more about how you're feeling. 💬"
