import pickle
from flask import Flask,request,jsonify,render_template
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
application = Flask(__name__)
app=application

## import ridge regressor and standard scale pickle
ridge_model=pickle.load(open('models/ridge.pkl','rb'))
standard_scale=pickle.load(open('models/scaler.pkl','rb'))
## these pickle files were created from model training last line
## Pickling is the process of converting a Python object into a byte stream so it can be saved to a file or transferred over a network — later, it can be "unpickled" (restored back) into the original object.


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predictdata',methods=['GET','POST'])

def predict_datapoint():
    if request.method=='POST':
        # pass
        temperature = float(request.form.get('Temperature'))
        rh = float(request.form.get('RH'))
        ws = float(request.form.get('WS'))
        rain = float(request.form.get('Rain'))
        ffmc = float(request.form.get('FFMC'))
        dmc = float(request.form.get('DMC'))
        isi = float(request.form.get('ISI'))
        classes = float(request.form.get('Classes'))
        region = float(request.form.get('Region'))

        new_data_scaled=standard_scale.transform([[temperature,rh,ws,rain,ffmc,dmc,isi,classes,region]])
        result=ridge_model.predict(new_data_scaled)

        return render_template('home.html',results=result[0])
        # predict method of scikit-learn models always returns an array, even if you pass a single sample.
    else:
        return render_template('home.html')

if __name__ == '__main__':
    app.run(host="0.0.0.0")
