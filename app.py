import numpy as np
from flask import Flask, request, render_template
import pickle

flask_app = Flask(__name__)
model = pickle.load(open("model.pkl", "rb"))

@flask_app.route("/", methods = ["GET"])
def Home():
    return render_template("index.html")

@flask_app.route("/predict", methods = ["POST"])
def predict():
    vnline_n  = float(request.form['vnline_n'])
    veaseg_c  = int(request.form['veaseg_c'])
    vccdeu_n  = float(request.form['vccdeu_n'])
    ttkbco_n  = float(request.form['ttkbco_n'])
    vedadc_n  = float(request.form['vedadc_n'])
    vgamma_c  = int(request.form['vgamma_c'])
    arreglo = np.array([[vnline_n, veaseg_c, vccdeu_n, ttkbco_n, vedadc_n, vgamma_c]])
    prediction = model.predict(arreglo)

    if prediction == 1:
        return render_template('index.html',prediction_text = "El cliente 'No portó'")
    elif prediction == 0:
        return render_template('index.html',prediction_text = "El cliente 'Si portó'")

if __name__ == "__main__":
    flask_app.run(debug = True)