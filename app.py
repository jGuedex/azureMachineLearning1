import numpy as np
from flask import Flask, request, jsonify, render_template, url_for
import joblib

# Iniciando o serviço em Flask
app = Flask(__name__)
#model = pickle.load(open('randomForestRegressor.pkl','rb'))

classes = ['CLARO FLEX/APP CLARO FLEX/ERRO ATIVAÇÃO / MIGRAÇ',
'MOBILE/CM/Falha No Pool', 'OPERAÇÕES E INFRAESTRUTURA/Abend Job',
'SISTEMAS NETUNO-UNO-O2A-STUCK P3',
'SISTEMAS-EXTRACOES-CRM-ANALISE IDENTIFICACAO DE ORIGEM DO PROBLEMA',
'SISTEMAS-NET-FATURAMENTO-ABORT JOB IDENTIFICACAO DE ORIGEM DO PROBLEMA',
'SUPORTE STORAGE / BACKUP/BACKUP ERRO DE EXECUÇÃO',
'WPP NACIONAL/Erro Efetuar Retomada', 'WPP NACIONAL/Falha no Pool',
'outros']

# Principal de rota da API
# Neste caso, será utilizada um template simples apenas para facilitar as inferências
@app.route('/')
def home():
    return render_template('home.html')

# Rota destinada a realizar as predições
# Após a execução da predição, então será renderizado o template principal com
# o retorno da predição
@app.route('/predict',methods = ['POST'])
def predict():
    query = request.form.get("query")

    if query == '':
        exit("Nenhuma classe informada")

    model = joblib.load("pipeline6.joblib")

    class_number = model.predict([query])
    class_name = classes[class_number[0]]

    print("---> ", class_number[0])
    
    # Há uma lista ordenada decrescente contendo as predições
    # É escolhido o índice zero, já que é o que tem a maior confiança
    #result = prediction[0]

    #return render_template('home.html', prediction_text="AQI for Jaipur {}".format(prediction[0]))
    return render_template('home.html', prediction_text="Categoria sugerida: {}".format(class_name))

# Rota responsável por receber o questionamento feito pelo usuário
@app.route('/predict_api',methods=['POST'])
def predict_api():
    data = request.get_json(force=True)
    prediction = model.predict([np.array(list(data.values()))])

    output = prediction[0]
    return jsonify(output)

if __name__ == '__main__':
    app.run(debug=True)