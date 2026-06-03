import pandas as pd
import os

INPUT_FILE = r"C:\Machine Learning\RGR\processed_cars.csv"
OUTPUT_NORMAL = "test_normal_car.csv"
OUTPUT_ANOMALY = "test_anomaly_car.csv"

def save_for_inference(df_sample, filename):
    true_price = df_sample['price_usd'].values[0] if 'price_usd' in df_sample.columns else "Не найдена"
    print(f"--- Файл: {filename} ---")
    print(f"Истинная цена (для сравнения в отчете): ${true_price:,.2f}")
    
    df_to_predict = df_sample.drop(columns=['price_usd'], errors='ignore')
    
    df_to_predict.to_csv(filename, index=False)

try:
    print("Загрузка датасета...")
    df = pd.read_csv(INPUT_FILE)
    print(f"Всего строк в датасете: {len(df)}\n")

    normal_cars = df[
        (df['car_age'] >= 3) & (df['car_age'] <= 15) &      
        (df['odometer_value'] >= 20000) & (df['odometer_value'] <= 200000)
    ]
    
    if len(normal_cars) > 0:

        sample_normal = normal_cars.sample(n=1).reset_index(drop=True)
        save_for_inference(sample_normal, OUTPUT_NORMAL)
    else:
        print(" Не удалось найти 'нормальную' машину по строгим критериям. Беру полностью случайную.")
        sample_normal = df.sample(n=1).reset_index(drop=True)
        save_for_inference(sample_normal, OUTPUT_NORMAL)

    anomaly_cars = df[
        (df['odometer_value'] > 350000) |
        (df['car_age'] > 25)     
    ]
    
    if len(anomaly_cars) > 0:
        sample_anomaly = anomaly_cars.sample(n=1).reset_index(drop=True)
        save_for_inference(sample_anomaly, OUTPUT_ANOMALY)
    else:
        print("Аномалии по заданным жестким критериям не найдены!")
        print("Попробуйте смягчить условия в скрипте (например, odometer_value > 250000 или car_age > 20).")


except FileNotFoundError:
    print(f"Ошибка: Файл '{INPUT_FILE}' не найден. Проверьте путь.")
except Exception as e:
    print(f"Произошла ошибка: {e}")
    