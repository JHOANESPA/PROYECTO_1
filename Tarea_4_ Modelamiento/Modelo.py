import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from pathlib import Path

#MODELO 1: 

# 1. Cargar datos

current_dir = Path(__file__).resolve().parent  
file_path = current_dir.parent / "Tarea_3_ Preparación de los datos" / "datos_limpios.xlsx"
df = pd.read_excel(file_path)


# Variables
features_num = ["reassignment_count", "reopen_count", "sys_mod_count"]
features_cat = ["impact", "urgency", "priority", "category",
                "assignment_group", "knowledge"]

target = "made_sla"


# 2. Procesamiento de datos

# Crear variables dummy SOLO para categóricas
X_cat = pd.get_dummies(df[features_cat], drop_first=True)
X_num = df[features_num]

# Combinar numéricas y categóricas
X = pd.concat([X_num, X_cat], axis=1)

# Convertir todas las columnas a float32 para evitar errores en TensorFlow
X = X.astype("float32")

# Variable objetivo como entero
y = df[target].astype(int)


# 3. División de datos

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)


# 4. Capa de normalización dentro del modelo
norm = layers.Normalization()
norm.adapt(np.array(X_train))

# 5. Construcción del modelo
model = keras.Sequential([
    norm,
    layers.Dense(32, activation="relu"),
    layers.Dense(16, activation="relu"),
    layers.Dense(1, activation="sigmoid")
])

model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

# 6. Entrenamiento
history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=20, batch_size=32, verbose=1
)

# 7. Evaluación
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"📊 Exactitud en test: {accuracy:.2%}")

# 8. Guardar modelo y columnas
model.save("modelo1_sla_tf_norm.h5")
pd.Series(X.columns).to_csv("columnas_modelo1.csv", index=False)
print("✅ Modelo 1 y columnas guardadas")

# 9. Predicción de ejemplo
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

prob = model.predict(nuevo_proc, verbose=0)[0][0]
print(f"🔮 Probabilidad de romper SLA: {prob:.2%}")
