from flask import Flask, request, render_template
import pickle
import numpy as np
app = Flask(__name__)
model = pickle.load(open('model.sav', 'rb'))
@app.route('/')
def p():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    name = [float(x) for x in request.form.values()]
    A = model.predict_proba([name])
    r = "Probability of  Diabetes for this User is %0.2f"%A[0][1]
    return render_template('index.html',result1 = r)

if __name__ == "__main__":
    app.run(debug=True)
