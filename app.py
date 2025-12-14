from flask import Flask, render_template, request
from src.components.pipeline.predict_pipeline import PredictionPipeline, CustomData

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["GET", "POST"])
def predict():
    prediction = None
    probability = None

    if request.method == "POST":
        data = CustomData(
            gender=request.form["gender"],
            SeniorCitizen=int(request.form["SeniorCitizen"]),
            Partner=request.form["Partner"],
            Dependents=request.form["Dependents"],
            tenure=int(request.form["tenure"]),
            PhoneService=request.form["PhoneService"],
            MultipleLines=request.form["MultipleLines"],
            InternetService=request.form["InternetService"],
            OnlineSecurity=request.form["OnlineSecurity"],
            OnlineBackup=request.form["OnlineBackup"],
            DeviceProtection=request.form["DeviceProtection"],
            TechSupport=request.form["TechSupport"],
            StreamingTV=request.form["StreamingTV"],
            StreamingMovies=request.form["StreamingMovies"],
            Contract=request.form["Contract"],
            PaperlessBilling=request.form["PaperlessBilling"],
            PaymentMethod=request.form["PaymentMethod"],
            MonthlyCharges=float(request.form["MonthlyCharges"]),
            TotalCharges=float(request.form["TotalCharges"]),
        )

        df = data.get_data_as_data_frame()
        pipeline = PredictionPipeline()

        pred_value = pipeline.predict(df)          # 0 or 1
        probability = pipeline.predict_proba(df)   # %

        prediction = "Churn" if pred_value == 1 else "No Churn"

    return render_template(
        "predict.html",
        prediction=prediction,
        probability=probability
    )


if __name__ == "__main__":
    app.run(debug=True)
