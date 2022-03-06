
from distutils.log import debug
import numpy as np
from flask import Flask,render_template, request, jsonify
import pickle




app=Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    int_features = [int(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)

    output = round(prediction[0], 2)

    return render_template('index.html', prediction_text='Customer is profitable? $ {}'.format(output))

@app.route('/predict_api',methods=['POST'])
def predict_api():
    '''
    For direct API calls trought request
    '''
    data = request.get_json(force=True)
    prediction = model.predict([np.array(list(data.values()))])

    output = prediction[0]
    return jsonify(output)

if __name__ == "__main__":
    app.run(debug=True)

"""
@app.route("/", methods=['GET','POST'])
def df():
    if request.method=="POST":
        income=request.form["income"]
        prof_pred=df.profitability_prediction(data)
        print(prof_pred)
    return render_template("index.html")
    """

"""
@app.route("/sub")
def submit():
    #html -> .py
    if request.method == "POST":
        name=request.form["username"]
    
    #.py-> html
    return render_template("sub.html", n=name)
    """



if __name__=="__main__":
    app.run(debug=True)