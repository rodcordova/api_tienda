from flask import Flask, request, jsonify
import numpy as np
#from tensorflow.keras.models import load_model
#import joblib
from joblib import dump, load
import pandas as pd

def return_prediction(model, sample_json):
    Precio_Actual = sample_json['Precio Actual']
    Tasa_x_Compra = sample_json['Tasa x Compra']
    Cantidad = sample_json['Cantidad']

    feat_cols=['Precio Actual','Tasa x Compra', 'Cantidad']
    row = [Precio_Actual, Tasa_x_Compra, Cantidad]

    features = pd.DataFrame([row], columns = feat_cols)#nombre de columna feat_cols


    #calculo clase
    predicted_class_index =model.predict(features)[0]# el [0] tiee mas alta probabilidad
    tipo="Sin descuento" if predicted_class_index ==0 else "Con descuento"
    #calculo con que probabilidad
    prediction = model.predict_proba(features)
    value = 100 *round( prediction[0][predicted_class_index ],2)

    return {"class":tipo,
            "value":value
        }

app = Flask(__name__)

@app.route('/')
def index():
    return '<h1>La aplicación de Flask está funcionando</h1>'


model = load('model.joblib')# cargo el modelo

@app.route('/api/prediction', methods=['POST'])
def predict_flower():
    content = request.json
    results = return_prediction(model=model, sample_json = content)

    return jsonify(results)

if __name__ == '__main__':
    app.run()#debug=True) #reiniciará el servidor automáticamente 
