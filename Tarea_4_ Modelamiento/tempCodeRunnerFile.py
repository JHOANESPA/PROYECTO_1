import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from pathlib import Path
import matplotlib.pyplot as plt

# ==============================
# 1. Cargar datos
# ==============================
current_dir = Path(__file__).resolve().parent
file_path = current_dir.parent / "Tarea_3_ Preparación de los datos" / "datos_limpios.xlsx"
df = pd.read_excel(file_path)

features_num = ["reassignment_count", "reopen_count", "sys_mod_count"]
features_cat = ["impact", "urgency", "priority", "category", "assignment_group", "knowledge"]
target = "made_sla"

# ==============================
# Función auxiliar de preparación
# ==============================
def preparar_datos(df):
    X_cat = pd.get_dummies(df[features_cat], drop_first=True)
    X_num = df[features_num]
    X = pd.concat([X_num, X_cat], axis=1).astype("float32")
    y = df[target].astype(int)
    return X, y

# ==============================
# MODELO 1: test_size=0.2 + Normalization
# ==============================
X, y = preparar_datos(df)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

norm = layers.Normalization()
norm.adapt(np.array(X_train))

model1 = keras.Sequential([
    norm,
    layers.Dense(32, activation="relu"),
    layers.Dense(16, activation="relu"),
    layers.Dense(1, activation="sigmoid")
])

model1.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
history1 = model1.fit(X_train, y_train, validation_data=(X_test, y_test),
                      epochs=20, batch_size=32, verbose=1)
model1.save("modelo1_sla_tf_norm.h5")
pd.Series(X.columns).to_csv("columnas_modelo1.csv", index=False)

# ==============================
# MODELO 2: test_size=0.3 + más capas + Normalization
# ==============================
X, y = preparar_datos(df)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)

norm2 = layers.Normalization()
norm2.adapt(np.array(X_train))

model2 = keras.Sequential([
    norm2,
    layers.Dense(64, activation="relu"),
    layers.Dense(32, activation="relu"),
    layers.Dense(16, activation="relu"),
    layers.Dense(1, activation="sigmoid")
])

model2.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
history2 = model2.fit(X_train, y_train, validation_data=(X_test, y_test),
                      epochs=20, batch_size=32, verbose=1)
model2.save("modelo2_sla_tf_norm.h5")
pd.Series(X.columns).to_csv("columnas_modelo2.csv", index=False)

# ==============================
# MODELO 3: test_size=0.3 + sin Normalization (StandardScaler en numéricas)
# ==============================
X, y = preparar_datos(df)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)

scaler = StandardScaler()
X_train_scaled = X_train.copy()
X_test_scaled = X_test.copy()
X_train_scaled[features_num] = scaler.fit_transform(X_train[features_num])
X_test_scaled[features_num] = scaler.transform(X_test[features_num])

model3 = keras.Sequential([
    layers.Input(shape=(X_train_scaled.shape[1],)),
    layers.Dense(64, activation="relu"),
    layers.Dense(32, activation="relu"),
    layers.Dense(16, activation="relu"),
    layers.Dense(1, activation="sigmoid")
])

model3.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
history3 = model3.fit(X_train_scaled, y_train, validation_data=(X_test_scaled, y_test),
                      epochs=20, batch_size=32, verbose=1)
model3.save("modelo3_sla_tf_stdnum.h5")
pd.Series(X.columns).to_csv("columnas_modelo3.csv", index=False)

# ==============================
# GRAFICAR FUNCIONES DE PÉRDIDA
# ==============================
def plot_loss(history, titulo):
    plt.figure(figsize=(6,4))
    plt.plot(history.history['loss'], label='Entrenamiento')
    plt.plot(history.history['val_loss'], label='Validación')
    plt.title(f'Función de pérdida – {titulo}')
    plt.xlabel('Épocas')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.show()

# Graficas individuales
plot_loss(history1, "Modelo 1")
plot_loss(history2, "Modelo 2")
plot_loss(history3, "Modelo 3")

# Comparación conjunta
plt.figure(figsize=(7,5))
plt.plot(history1.history['loss'], label='Modelo 1')
plt.plot(history2.history['loss'], label='Modelo 2')
plt.plot(history3.history['loss'], label='Modelo 3')
plt.title('Comparación de pérdida (entrenamiento)')
plt.xlabel('Épocas')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(7,5))
plt.plot(history1.history['val_loss'], label='Modelo 1')
plt.plot(history2.history['val_loss'], label='Modelo 2')
plt.plot(history3.history['val_loss'], label='Modelo 3')
plt.title('Comparación de pérdida (validación)')
plt.xlabel('Épocas')
plt.ylabel('Val_loss')
plt.legend()
plt.grid(True)
plt.show()


# Extraer exactitud final de cada modelo
acc1 = history1.history['val_accuracy'][-1] * 100
acc2 = history2.history['val_accuracy'][-1] * 100
acc3 = history3.history['val_accuracy'][-1] * 100

# Comparación de pérdida (entrenamiento)
plt.figure(figsize=(7,5))
plt.plot(history1.history['loss'], label=f'Modelo 1 ({acc1:.1f}% acc)')
plt.plot(history2.history['loss'], label=f'Modelo 2 ({acc2:.1f}% acc)')
plt.plot(history3.history['loss'], label=f'Modelo 3 ({acc3:.1f}% acc)')
plt.title('Comparación de pérdida (entrenamiento)')
plt.xlabel('Épocas')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.show()

# Comparación de pérdida (validación)
plt.figure(figsize=(7,5))
plt.plot(history1.history['val_loss'], label=f'Modelo 1 ({acc1:.1f}% acc)')
plt.plot(history2.history['val_loss'], label=f'Modelo 2 ({acc2:.1f}% acc)')
plt.plot(history3.history['val_loss'], label=f'Modelo 3 ({acc3:.1f}% acc)')
plt.title('Comparación de pérdida (validación)')
plt.xlabel('Épocas')
plt.ylabel('Val_loss')
plt.legend()
plt.grid(True)
plt.show()

