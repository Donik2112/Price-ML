from flask import Flask, request, render_template
import joblib
import pandas as pd
import os
import matplotlib.pyplot as plt
import time

app = Flask(__name__)

base_dir = os.path.dirname(os.path.abspath(__file__))

# Пути к моделям
classifier_path = os.path.join(base_dir, 'models', 'car_price_classifier.pkl')
regressor_path = os.path.join(base_dir, 'models', 'car_price_regressor.pkl')

# Загрузка моделей
try:
    classifier = joblib.load(classifier_path)
    print(f"Классификационная модель успешно загружена из {classifier_path}")
except Exception as e:
    print(f"Ошибка при загрузке классификационной модели: {e}")
    classifier = None

try:
    model_reg = joblib.load(regressor_path)
    print(f"Регрессионная модель успешно загружена из {regressor_path}")
except Exception as e:
    print(f"Ошибка загрузки регрессионной модели: {e}")
    model_reg = None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if classifier is None:
        return render_template('index.html', prediction="Ошибка: модель не загружена")

    try:
        data = {
            'model_year': int(request.form['model_year']),
            'milage': float(request.form['milage']),
            'engine_power': float(request.form['engine_power']),
            'brand': request.form['brand'],
            'model': request.form['model'],
            'fuel_type': request.form['fuel_type'],
            'transmission': request.form['transmission'],
            'ext_col': request.form['ext_col'],
            'int_col': request.form['int_col'],
            'accident': request.form['accident']
        }

        df = pd.DataFrame([data])
        prediction = classifier.predict(df)[0]
        result = "Дорогой автомобиль (1)" if prediction == 1 else "Бюджетный автомобиль (0)"
        return render_template('index.html', prediction=result)

    except Exception as e:
        return render_template('index.html', prediction=f"Ошибка при предсказании: {str(e)}")

@app.route('/predict_price', methods=['POST'])
def predict_price():
    if model_reg is None:
        return render_template('index.html', prediction="Ошибка: регрессионная модель не загружена")

    try:
        data = {
            'model_year': int(request.form['model_year']),
            'milage': float(request.form['milage']),
            'engine_power': float(request.form['engine_power']),
            'brand': request.form['brand'],
            'model': request.form['model'],
            'fuel_type': request.form['fuel_type'],
            'transmission': request.form['transmission'],
            'ext_col': request.form['ext_col'],
            'int_col': request.form['int_col'],
            'accident': request.form['accident']
        }

        df = pd.DataFrame([data])
        price_prediction = model_reg.predict(df)[0]

        return render_template('index.html', prediction=f"Оценочная цена автомобиля: ${price_prediction:,.0f}")

    except Exception as e:
        return render_template('index.html', prediction=f"Ошибка: {str(e)}")

if __name__ == '__main__':
    app.run(debug=True)
