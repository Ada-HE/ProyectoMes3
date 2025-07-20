from flask import Flask, request, render_template, jsonify
import joblib
import pandas as pd
import numpy as np
import logging

app = Flask(__name__)

# Configurar el registro
logging.basicConfig(level=logging.DEBUG)

# Cargar el modelo, el scaler, el label encoder y el ordinal encoder
model = joblib.load("random_forest_model.pkl")  # Modelo de Random Forest
scaler = joblib.load("scaler.pkl")  # Scaler usado para escalar las variables
label_encoder = joblib.load("label_encoder_estado_cita.pkl")  # LabelEncoder para estado_cita
ordinal_encoder = joblib.load("ordinal_encoder.pkl")  # OrdinalEncoder para estado_tratamiento
selected_features = joblib.load("selected_features.pkl")  # Características seleccionadas

app.logger.debug("Modelo, scaler, label encoder y ordinal encoder cargados correctamente.")
app.logger.debug(f"Ordinal encoder categories: {ordinal_encoder.categories_}")
app.logger.debug(f"Scaler feature names: {scaler.get_feature_names_out()}")
app.logger.debug(f"Selected features: {selected_features}")

@app.route('/')
def home():
    return render_template('formulario.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Obtener los datos enviados en el request
        edad = float(request.form['edad'])
        citas_totales = float(request.form['citas_totales'])
        citas_asistidas = float(request.form['citas_asistidas'])
        estado_tratamiento = request.form['estado_tratamiento']  # Mantener en minúsculas
        monto_ultimo_pago = float(request.form['monto_ultimo_pago'])
        dias_entre_citas = float(request.form['dias_entre_citas'])

        app.logger.debug(f"Datos recibidos: edad={edad}, citas_totales={citas_totales}, "
                        f"citas_asistidas={citas_asistidas}, estado_tratamiento={estado_tratamiento}, "
                        f"monto_ultimo_pago={monto_ultimo_pago}, dias_entre_citas={dias_entre_citas}")

        # Convertir datos categóricos a numéricos usando el ordinal encoder
        cat_data = [[estado_tratamiento]]  # Formato 2D para el encoder
        encoded_cat_data = ordinal_encoder.transform(cat_data)
        app.logger.debug(f"Valores codificados: {encoded_cat_data}")

        # Crear un DataFrame con los datos transformados
        data_df = pd.DataFrame({
            'edad': [edad],
            'citas_totales': [citas_totales],
            'citas_asistidas': [citas_asistidas],
            'estado_tratamiento': [encoded_cat_data[0][0]],
            'monto_ultimo_pago': [monto_ultimo_pago],
            'dias_entre_citas': [dias_entre_citas]
        }, columns=selected_features)
        app.logger.debug(f"DataFrame antes de escalar: {data_df}")

        # Escalar los datos usando el scaler cargado
        data_df_scaled = pd.DataFrame(scaler.transform(data_df), columns=data_df.columns)
        app.logger.debug(f"DataFrame escalado: {data_df_scaled}")

        # Realizar predicciones con el modelo
        prediction = model.predict(data_df_scaled)
        predicted_label = label_encoder.inverse_transform(prediction)[0]
        app.logger.debug(f"Predicción: {predicted_label}")

        return jsonify({'estado_cita': predicted_label})
    except Exception as e:
        app.logger.error(f"Error en la predicción: {str(e)}")
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)