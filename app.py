from flask import Flask, request, render_template
import pickle

app = Flask(__name__)

# Load trained model
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None
    confidence = None

    if request.method == "POST":
        text = request.form["text"]

        transformed = vectorizer.transform([text])
        pred = model.predict(transformed)[0]
        prob = model.predict_proba(transformed).max()

        label = "Fake/Scam" if pred == 1 else "Real/Legit"

        prediction = label
        confidence = round(prob * 100, 2)

    return render_template("index.html",
                           prediction=prediction,
                           confidence=confidence)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)


