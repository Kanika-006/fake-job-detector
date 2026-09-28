import joblib
import os


# Get project root directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Load trained model and vectorizer
model = joblib.load(
    os.path.join(BASE_DIR, "models", "fake_job_svm.pkl")
)

vectorizer = joblib.load(
    os.path.join(BASE_DIR, "models", "tfidf_vectorizer.pkl")
)


def predict_job(text):
    """
    Predict whether a job posting is potentially fraudulent.
    """

    # Convert text into TF-IDF features
    text_vector = vectorizer.transform([text])

    # SVM prediction
    prediction = model.predict(text_vector)[0]

    # SVM decision score
    decision_score = model.decision_function(text_vector)[0]

    if prediction == 1:
        result = "Potentially Fraudulent"
    else:
        result = "Likely Genuine"

    return result, decision_score

def detect_suspicious_signals(text):
    text_lower = text.lower()

    signals = []

    # 1. Payment / money requests
    payment_keywords = [
        "pay a fee",
        "registration fee",
        "application fee",
        "processing fee",
        "deposit",
        "pay us",
        "send money",
        "wire money"
    ]

    if any(keyword in text_lower for keyword in payment_keywords):
        signals.append(
            ("Payment request", "The posting appears to request money or a fee.")
        )

    # 2. Personal / financial information
    sensitive_keywords = [
        "bank account",
        "bank details",
        "credit card",
        "debit card",
        "social security",
        "passport number",
        "send your id"
    ]

    if any(keyword in text_lower for keyword in sensitive_keywords):
        signals.append(
            ("Sensitive information request",
             "The posting appears to request personal or financial information.")
        )

    # 3. Urgency / pressure
    urgency_keywords = [
        "act now",
        "urgent",
        "immediately",
        "limited time",
        "apply today",
        "hurry",
        "quickly"
    ]

    if any(keyword in text_lower for keyword in urgency_keywords):
        signals.append(
            ("Urgency language",
             "The posting uses language that creates pressure to act quickly.")
        )

    # 4. Work-from-home / easy-money claims
    easy_money_keywords = [
        "work from home",
        "make money fast",
        "earn money quickly",
        "easy money",
        "guaranteed income",
        "no experience required"
    ]

    if any(keyword in text_lower for keyword in easy_money_keywords):
        signals.append(
            ("Easy-income / work-from-home language",
             "The posting contains claims commonly associated with high-risk job offers.")
        )

    # 5. External links
    if "http://" in text_lower or "https://" in text_lower or "www." in text_lower:
        signals.append(
            ("External link",
             "The posting contains an external web link.")
        )

    # 6. Very short posting
    word_count = len(text.split())

    if word_count < 40:
        signals.append(
            ("Very short posting",
             "The job posting contains unusually little information.")
        )

    return signals

if __name__ == "__main__":
    sample_job = """
    We are looking for a software engineer to join our team.
    The candidate should have experience with Python, SQL,
    machine learning and data analysis.
    This is a full-time position with competitive salary and benefits.
    """

    result, score = predict_job(sample_job)

    print("Prediction:", result)
    print("Decision score:", round(score, 3))

    print("\nSuspicious signals:")

    signals = detect_suspicious_signals(sample_job)

    if signals:
        for name, explanation in signals:
            print("-", name, ":", explanation)
    else:
        print("No predefined suspicious signals detected.")