
from flask import Flask, render_template, request
import pickle

app = Flask(__name__)

model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None
    confidence = None

    if request.method == "POST":
        news = request.form["news"]

        transformed = vectorizer.transform([news])
        result = model.predict(transformed)[0]
        probability = model.predict_proba(transformed)[0]

        if result == 1:
            prediction = "REAL NEWS"
            confidence = round(probability[1] * 100, 2)
        else:
            prediction = "FAKE NEWS"
            confidence = round(probability[0] * 100, 2)

    return render_template(
        "index.html",
        prediction=prediction,
        confidence=confidence
    )

if __name__ == "__main__":
    app.run(debug=True)
