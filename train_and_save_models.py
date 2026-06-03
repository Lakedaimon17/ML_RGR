import os
import pandas as pd
import numpy as np
import joblib
import warnings

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor, StackingRegressor
from catboost import CatBoostRegressor
from keras.models import Sequential
from keras.layers import Dense, Dropout

warnings.filterwarnings('ignore')


# 1. НАСТРОЙКИ И ЗАГРУЗКА ДАННЫХ
DATA_PATH = 'data\processed_cars.csv' 
MODELS_DIR = 'models'
RANDOM_STATE = 42

print("Загрузка данных...")
if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"Файл {DATA_PATH} не найден. Проверьте путь к датасету.")

df = pd.read_csv(DATA_PATH)

target_col = 'price_usd'
X = df.drop(columns=[target_col])
y = df[target_col]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)


# 2. ПРЕДОБРАБОТКА (Масштабирование)
print("Масштабирование данных...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

os.makedirs(MODELS_DIR, exist_ok=True)

joblib.dump(scaler, os.path.join(MODELS_DIR, 'scaler.joblib'))
print(f"Scaler сохранен в {MODELS_DIR}/scaler.joblib")


# 3. ОБУЧЕНИЕ И СОХРАНЕНИЕ МОДЕЛЕЙ
print("\nНачало обучения и сохранения моделей...")

# --- ML1: Классическая модель (Ridge) ---
print("1. Обучение Ridge Regression...")
model_ridge = Ridge(alpha=1.0, random_state=RANDOM_STATE)
model_ridge.fit(X_train_scaled, y_train)
joblib.dump(model_ridge, os.path.join(MODELS_DIR, 'ridge_model.joblib'))

# --- ML2: Ансамблевая модель (Gradient Boosting) ---
print("2. Обучение Gradient Boosting...")
model_gb = GradientBoostingRegressor(
    n_estimators=100, learning_rate=0.1, max_depth=5, random_state=RANDOM_STATE
)
model_gb.fit(X_train_scaled, y_train)
joblib.dump(model_gb, os.path.join(MODELS_DIR, 'gb_model.joblib'))

# --- ML3: Продвинутый бустинг (CatBoost) ---
print("3. Обучение CatBoost...")
model_catboost = CatBoostRegressor(
    iterations=100, learning_rate=0.1, depth=6, verbose=0, random_state=RANDOM_STATE
)
model_catboost.fit(X_train_scaled, y_train)

model_catboost.save_model(os.path.join(MODELS_DIR, 'catboost_model.cbm'))

# --- ML4: Бэггинг (Random Forest) ---
print("4. Обучение Random Forest...")
model_rf = RandomForestRegressor(
    n_estimators=100, max_depth=10, random_state=RANDOM_STATE
)
model_rf.fit(X_train_scaled, y_train)

joblib.dump(model_rf, os.path.join(MODELS_DIR, 'rf_model.joblib'))

# --- ML5: Стекинг (Stacking) ---
print("5. Обучение Stacking Regressor...")

base_estimators = [
    ('ridge', Ridge(alpha=1.0, random_state=RANDOM_STATE)),
    ('rf', RandomForestRegressor(n_estimators=30, max_depth=10, random_state=RANDOM_STATE))
]

final_estimator = Ridge(alpha=1.0)
model_stacking = StackingRegressor(
    estimators=base_estimators, 
    final_estimator=final_estimator, 
    cv=3
)
model_stacking.fit(X_train_scaled, y_train)
joblib.dump(model_stacking, os.path.join(MODELS_DIR, 'stacking_model.joblib'))

# --- ML6: Глубокая полносвязная нейронная сеть (Keras) ---
print("6. Обучение Keras FCNN...")
input_dim = X_train_scaled.shape[1]
model_keras = Sequential([
    Dense(128, activation='relu', input_shape=(input_dim,)),
    Dropout(0.2),
    Dense(64, activation='relu'),
    Dropout(0.2),
    Dense(32, activation='relu'),
    Dense(1) 
])
model_keras.compile(optimizer='adam', loss='mse')
model_keras.fit(X_train_scaled, y_train, epochs=50, batch_size=64, verbose=0)

model_keras.save(os.path.join(MODELS_DIR, 'keras_model.keras'))

print("\nВсе 6 моделей и скалер успешно обучены и сохранены в папку 'models/'!")


# 4. ПРОВЕРКА
print("\n--- Быстрая проверка качества на тестовой выборке ---")
from sklearn.metrics import r2_score, mean_absolute_error

models_to_check = {
    'Ridge': model_ridge,
    'GradientBoosting': model_gb,
    'CatBoost': model_catboost,
    'RandomForest': model_rf,
    'Stacking': model_stacking
}

for name, model in models_to_check.items():
    preds = model.predict(X_test_scaled)
    r2 = r2_score(y_test, preds)
    mae = mean_absolute_error(y_test, preds)
    print(f"{name:<15} | R²: {r2:.4f} | MAE: {mae:,.2f}")

preds_keras = model_keras.predict(X_test_scaled, verbose=0).flatten()
r2_keras = r2_score(y_test, preds_keras)
mae_keras = mean_absolute_error(y_test, preds_keras)
print(f"{'Keras FCNN':<15} | R²: {r2_keras:.4f} | MAE: {mae_keras:,.2f}")
