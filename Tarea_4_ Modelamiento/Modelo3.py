# =======================================================================
# MODELO 3 con test_size 0.3 SIN capa de normalización (escala numéricas)
# =======================================================================
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from pathlib import Path

# 1. Cargar datos
current_dir = Path(__file__).resolve().parent
file_path = current_dir.parent / "Tarea_3_ Preparación de los datos" / "datos_limpios.xlsx"
df = pd.read_excel(file_path)

# Variables
features_num = ["reassignment_count", "reopen_count", "sys_mod_count"]
features_cat = ["impact", "urgency", "priority", "category",
                "assignment_group", "knowledge"]
target = "made_sla"

# 2. Procesamiento de datos (igual que en tu MODELO 2)
X_cat = pd.get_dummies(df[features_cat], drop_first=True)
X_num = df[features_num]
X = pd.concat([X_num, X_cat], axis=1).astype("float32")
y = df[target].astype(int)

# 3. División de datos (igual)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# 4. Escalado externo SOLO de las columnas numéricas (reemplaza a la capa Normalization)
scaler = StandardScaler()
X_train_scaled = X_train.copy()
X_test_scaled = X_test.copy()
X_train_scaled[features_num] = scaler.fit_transform(X_train[features_num])
X_test_scaled[features_num] = scaler.transform(X_test[features_num])

# 5. Construcción del modelo (igual pero SIN 'norm' al inicio)
model = keras.Sequential([
    layers.Input(shape=(X_train_scaled.shape[1],)),
    layers.Dense(64, activation="relu"),
    layers.Dense(32, activation="relu"),
    layers.Dense(16, activation="relu"),
    layers.Dense(1, activation="sigmoid")
])

model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

# 6. Entrenamiento
history = model.fit(
    X_train_scaled, y_train,
    validation_data=(X_test_scaled, y_test),
    epochs=20, batch_size=32, verbose=1
)

# 7. Evaluación
loss, accuracy = model.evaluate(X_test_scaled, y_test, verbose=0)
print(f"📊 Exactitud en test MODELO 3: {accuracy:.2%}")

# 8. Guardar modelo y columnas (igual)
model.save("modelo3_sla_tf_stdnum.h5")
pd.Series(X.columns).to_csv("columnas_modelo3.csv", index=False)
print("✅ Modelo 3 y columnas guardadas")

# 9. Predicción de ejemplo (igual que en tu MODELO 2, con el mismo one-hot y alineación)
nuevo_ticket = pd.DataFrame([{
    "reassignment_count": 3,
    "reopen_count": 0,
    "sys_mod_count": 10,
    "impact": "1 - High",
    "urgency": "1 - High",
    "priority": "1 - Critical",
    "category": "Category 40",
    "assignment_group": "Group 56",
    "knowledge": True
}])

nuevo_cat = pd.get_dummies(nuevo_ticket[features_cat], drop_first=True)
nuevo_proc = pd.concat([nuevo_ticket[features_num], nuevo_cat], axis=1).astype("float32")
for col in X.columns:
    if col not in nuevo_proc.columns:
        nuevo_proc[col] = 0
nuevo_proc = nuevo_proc[X.columns]

# Escalar SOLO las numéricas con el mismo scaler (único añadido en predicción)
nuevo_proc[features_num] = scaler.transform(nuevo_proc[features_num])

prob = model.predict(nuevo_proc, verbose=0)[0][0]
print(f"🔮 Probabilidad de romper SLA MODELO 3: {prob:.2%}")