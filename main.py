from fastapi import FastAPI, File, UploadFile, Response
from fastapi.responses import HTMLResponse
import pandas as pd
import numpy as np
import pickle
from io import BytesIO

app = FastAPI()

# Cargar modelo y datos estáticos al iniciar la aplicación
model1 = None
df1 = None

@app.on_event("startup")
def load_assets():
    global model1, df1
    with open('mejor_modelo_julio_11-8-22', 'rb') as f:
        model1 = pickle.load(f)
    df1 = pd.read_excel('df1.xlsx')

@app.get("/", response_class=HTMLResponse)
def read_root():
    return """
    <html>
        <head>
            <title>API de Predicción</title>
            <style>
                body {
                    font-family: Arial, sans-serif;
                    margin: 40px;
                    background-color: #f0f2f5;
                }
                .container {
                    max-width: 800px;
                    margin: 0 auto;
                    padding: 20px;
                    background-color: white;
                    border-radius: 8px;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                }
                h1 {
                    color: #1a73e8;
                    border-bottom: 2px solid #1a73e8;
                    padding-bottom: 10px;
                }
                p {
                    line-height: 1.6;
                    color: #333;
                }
            </style>
        </head>
        <body>
            <div class="container">
                <h1>API Funcional 🚀</h1>
                <p>Bienvenido al sistema de predicción de riesgo.</p>
                <p>Usa el endpoint <code>/model_predict</code> via POST con un archivo XLSX para obtener predicciones.</p>
                <p>Estado del servicio: <span style="color: green; font-weight: bold;">Operativo</span></p>
            </div>
        </body>
    </html>
    """

@app.post("/model_predict")
async def model_predict(file: UploadFile = File(...)):
    content = await file.read()
    data = pd.read_excel(BytesIO(content))
    
    predacierta = model1.predict_proba(data)
    predacierta = [p[1] for p in predacierta]
    prediccion_acierta = pd.Series(predacierta, name='prob_90')
    
    prediccionrf = pd.concat([df1, prediccion_acierta], axis=1)
    
    risk_class = []
    for row in prediccionrf['prob_90']:
        if row > 0.1000039:
            risk_class.append('Extremo')
        elif row > 0.0500117:
            risk_class.append('Muy Alto')
        elif row > 0.0200018:
            risk_class.append('Alto')
        elif row > 0.0060008:
            risk_class.append('Moderado')
        elif row > 0.0020001:
            risk_class.append('Bajo')
        else:
            risk_class.append('Muy Bajo')
    
    prediccionrf['risk_class'] = risk_class
    prediccionrf['Score'] = 99.9 - prediccionrf['prob_90'] * 100
    
    csv_data = prediccionrf.to_csv(index=False)
    
    return Response(
        content=csv_data,
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=predicciones.csv"}
    )