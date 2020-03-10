from flask import Flask, render_template, request
from sklearn.externals import joblib
import pandas as pd
import numpy as np

app = Flask(__name__)

mul_reg = open("multiple_regression_model.pkl", "rb")
ml_model = joblib.load(mul_reg)

@app.route("/")
def home():
    return render_template('home.html')

@app.route("/predict", methods=['GET', 'POST'])
def predict():
    print("I was here 1")
    if request.method == 'POST':
        print(request.form.get('OutCome'))
        try:
           
            Pregnancies = float(request.form['Pregnancies'])
            Glucose = float(request.form['Glucose'])
            BloodPressure = float(request.form['BloodPressure'])
            SkinThickness=float(request.form['SkinThickness'])
            Insulin=      float(request.form['Insulin'])
            BMI=float(request.form['BMI'])
            DiabetesPedigreeFunction=float(request.form['DiabetesPedigreeFunction'])
            Age=float(request.form['Age'])
            pred_args = [  Pregnancies, Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age ]
            pred_args_arr = np.array(pred_args)
            pred_args_arr = pred_args_arr.reshape(1, -1)
            mul_reg = open("multiple_regression_model.pkl", "rb")
            ml_model = joblib.load(mul_reg)
            model_prediction = ml_model.predict(pred_args_arr)
            model_prediction = round(float(model_prediction), 2)
        except ValueError:
            return "Please check if the values are entered correctly"
    return render_template('predict.html', prediction = model_prediction)


if __name__ == "__main__":
    app.run()
