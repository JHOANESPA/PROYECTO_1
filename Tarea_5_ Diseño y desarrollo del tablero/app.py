import io
import base64
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.express as px
from dash import Dash, dcc, html, dash_table, Input, Output, State, callback_context

import tensorflow as tf
import keras

# =========================
# Config / carga de artefactos
# =========================
HERE = Path(__file__).resolve().parent
MODEL_PATH = HERE / "modelo1_sla_tf_norm.h5"
COLS_PATH = HERE / "columnas_modelo1.csv"

model = None
feature_columns = None
load_msg = []

try:
    model = keras.models.load_model(MODEL_PATH)
    load_msg.append("✅ Modelo cargado")
except Exception as e:
    load_msg.append(f"⚠️ No se pudo cargar el modelo: {e}")

try:
    feature_columns = pd.read_csv(COLS_PATH, header=None).iloc[:, 0].tolist()
    load_msg.append("✅ Columnas de entrenamiento cargadas")
except Exception as e:
    load_msg.append(f"⚠️ No se pudo cargar columnas_modelo1.csv: {e}")

# Variables del usuario (tu diseño)
FEATURES_NUM = ["reassignment_count", "reopen_count", "sys_mod_count"]
FEATURES_CAT = ["impact", "urgency", "priority", "category", "assignment_group", "knowledge"]
TARGET = "made_sla"

# Opciones por defecto (ajústalas si tus datos tienen otras etiquetas)
IMPACT_OPTS = ["1 - High", "2 - Medium", "3 - Low"]
URGENCY_OPTS = ["1 - High", "2 - Medium", "3 - Low"]
PRIORITY_OPTS = ["1 - Critical", "2 - High", "3 - Moderate", "4 - Low"]
KNOWLEDGE_OPTS = [True, False]

# =========================
# Utilidades
# =========================
def preprocess_single_row(row_dict: dict, feature_columns: list) -> pd.DataFrame:
    """
    Toma un dict con las 9 entradas (3 num + 6 cat),
    crea dummies SOLO para categóricas, alinea columnas con feature_columns.
    """
    df_row = pd.DataFrame([row_dict])

    # cast tipos
    df_row["reassignment_count"] = pd.to_numeric(df_row["reassignment_count"], errors="coerce").fillna(0)
    df_row["reopen_count"] = pd.to_numeric(df_row["reopen_count"], errors="coerce").fillna(0)
    df_row["sys_mod_count"] = pd.to_numeric(df_row["sys_mod_count"], errors="coerce").fillna(0)

    # dummies
    X_cat = pd.get_dummies(df_row[FEATURES_CAT], drop_first=True)
    X_num = df_row[FEATURES_NUM]
    X = pd.concat([X_num, X_cat], axis=1).astype("float32")

    # alinear a columnas del entrenamiento
    for col in feature_columns:
        if col not in X.columns:
            X[col] = 0.0
    # mismo orden
    X = X[feature_columns].astype("float32")
    return X

def parse_upload(contents, filename):
    """
    Lee .csv o .xlsx desde el Upload de Dash y regresa un DataFrame.
    """
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    if filename.lower().endswith(".csv"):
        return pd.read_csv(io.BytesIO(decoded))
    if filename.lower().endswith(".xlsx") or filename.lower().endswith(".xls"):
        return pd.read_excel(io.BytesIO(decoded))
    raise ValueError("Formato no soportado. Sube .csv o .xlsx")

def safe_rate_by(df, col):
    # Calcula % de made_sla=1 por categoría si existen las columnas
    if TARGET in df.columns and col in df.columns:
        tmp = (df
               .groupby(col)[TARGET]
               .mean()
               .reset_index(name="tasa_SLA"))
        return tmp
    return pd.DataFrame()

# =========================
# App y Layout
# =========================
app = Dash(__name__)
app.title = "SLA Predictor"

badge_model = html.Div(
    [html.Div(m) for m in load_msg],
    style={"fontSize": "12px", "color": "#555", "lineHeight": "18px", "marginBottom": "8px"}
)

form_card = html.Div([
    html.H3("🧾 Ingresar datos del ticket"),
    html.Div([
        html.Div([
            html.Label("reassignment_count"),
            dcc.Input(id="inp-reassign", type="number", value=3, min=0, step=1, debounce=True),
        ], className="col"),
        html.Div([
            html.Label("reopen_count"),
            dcc.Input(id="inp-reopen", type="number", value=0, min=0, step=1, debounce=True),
        ], className="col"),
        html.Div([
            html.Label("sys_mod_count"),
            dcc.Input(id="inp-sysmod", type="number", value=10, min=0, step=1, debounce=True),
        ], className="col"),
    ], className="row"),

    html.Div([
        html.Div([
            html.Label("impact"),
            dcc.Dropdown(IMPACT_OPTS, IMPACT_OPTS[0], id="inp-impact", clearable=False),
        ], className="col"),
        html.Div([
            html.Label("urgency"),
            dcc.Dropdown(URGENCY_OPTS, URGENCY_OPTS[0], id="inp-urgency", clearable=False),
        ], className="col"),
        html.Div([
            html.Label("priority"),
            dcc.Dropdown(PRIORITY_OPTS, PRIORITY_OPTS[0], id="inp-priority", clearable=False),
        ], className="col"),
    ], className="row"),

    html.Div([
        html.Div([
            html.Label("category (texto exacto)"),
            dcc.Input(id="inp-category", type="text", value="Category 40"),
        ], className="col"),
        html.Div([
            html.Label("assignment_group (texto exacto)"),
            dcc.Input(id="inp-group", type="text", value="Group 56"),
        ], className="col"),
        html.Div([
            html.Label("knowledge"),
            dcc.Dropdown(KNOWLEDGE_OPTS, KNOWLEDGE_OPTS[0], id="inp-knowledge", clearable=False),
        ], className="col"),
    ], className="row"),

    html.Button("Calcular probabilidad", id="btn-predict", n_clicks=0, className="btn"),
    html.Div(id="pred-output", style={"fontSize": "20px", "marginTop": "10px", "fontWeight": "bold"}),
], className="card")

upload_card = html.Div([
    html.H3("📤 Subir datos (CSV/Excel) para visualizaciones"),
    dcc.Upload(
        id='upload-data',
        children=html.Div(['Arrastra y suelta o ', html.A('selecciona archivo')]),
        style={
            'width': '100%', 'height': '80px', 'lineHeight': '80px',
            'borderWidth': '1px', 'borderStyle': 'dashed',
            'borderRadius': '6px', 'textAlign': 'center', 'margin': '10px 0'
        },
        multiple=False
    ),
    html.Div(id='upload-info', style={"fontSize": "12px", "color": "#555"}),
    html.Div([
        dcc.Graph(id="fig-impact"),
        dcc.Graph(id="fig-urgency"),
        dcc.Graph(id="fig-priority"),
    ])
], className="card")

help_card = html.Div([
    html.H3("ℹ️ Instrucciones"),
    html.Ul([
        html.Li("Ingresa valores y presiona “Calcular probabilidad”."),
        html.Li("La probabilidad mostrada es de ROMPER el SLA (salida sigmoide del modelo)."),
        html.Li("Para gráficos, sube un CSV/XLSX con columnas al menos: "
                "`impact`, `urgency`, `priority`, `made_sla` y (opcional) las numéricas."),
        html.Li("Los textos de `category` y `assignment_group` deben coincidir con los del entrenamiento para activar sus dummies."),
    ])
], className="card")

app.layout = html.Div([
    html.H1("Dashboard – Predicción de SLA"),
    badge_model,
    html.Div([
        html.Div(form_card, className="col col-40"),
        html.Div(upload_card, className="col col-60"),
    ], className="row"),
    help_card,

    # Estado oculto para almacenar el DF subido
    dcc.Store(id="store-df")
], className="container")

# =========================
# Callbacks
# =========================
@app.callback(
    Output("pred-output", "children"),
    Input("btn-predict", "n_clicks"),
    State("inp-reassign", "value"),
    State("inp-reopen", "value"),
    State("inp-sysmod", "value"),
    State("inp-impact", "value"),
    State("inp-urgency", "value"),
    State("inp-priority", "value"),
    State("inp-category", "value"),
    State("inp-group", "value"),
    State("inp-knowledge", "value"),
    prevent_initial_call=True
)
def make_prediction(nc, reassignment, reopen, sysmod, impact, urgency, priority, category, group, knowledge):
    if model is None or feature_columns is None:
        return "⚠️ No hay modelo/columnas cargadas."

    row = {
        "reassignment_count": reassignment or 0,
        "reopen_count": reopen or 0,
        "sys_mod_count": sysmod or 0,
        "impact": impact,
        "urgency": urgency,
        "priority": priority,
        "category": category or "",
        "assignment_group": group or "",
        "knowledge": bool(knowledge),
    }
    X = preprocess_single_row(row, feature_columns)
    prob = float(model.predict(X, verbose=0)[0][0])  # prob de romper SLA
    pct = f"{prob*100:.2f}%"

    # color semáforo
    if prob < 0.33:
        color = "#1a7f37"  # verde
        txt = "Riesgo BAJO de romper SLA"
    elif prob < 0.66:
        color = "#b7791f"  # amarillo
        txt = "Riesgo MEDIO de romper SLA"
    else:
        color = "#b42318"  # rojo
        txt = "Riesgo ALTO de romper SLA"

    return html.Span([f"Probabilidad: {pct} – ", txt], style={"color": color})

@app.callback(
    Output("store-df", "data"),
    Output("upload-info", "children"),
    Input("upload-data", "contents"),
    State("upload-data", "filename"),
    prevent_initial_call=True
)
def handle_upload(contents, filename):
    if contents is None:
        return None, ""
    try:
        df = parse_upload(contents, filename)
        info = f"Archivo cargado: {filename} – filas: {len(df):,}"
        # Reducir tamaño para almacenarlo en Store (opcional)
        return df.to_json(date_format='iso', orient='split'), info
    except Exception as e:
        return None, f"Error al leer el archivo: {e}"

@app.callback(
    Output("fig-impact", "figure"),
    Output("fig-urgency", "figure"),
    Output("fig-priority", "figure"),
    Input("store-df", "data")
)
def make_charts(df_json):
    empty_fig = px.scatter(title="Sube un archivo para ver gráficos")
    if not df_json:
        return empty_fig, empty_fig, empty_fig

    df = pd.read_json(df_json, orient='split')

    f1 = safe_rate_by(df, "impact")
    f2 = safe_rate_by(df, "urgency")
    f3 = safe_rate_by(df, "priority")

    fig1 = px.bar(f1, x="impact", y="tasa_SLA", title="Tasa de cumplimiento SLA por Impact")
    fig2 = px.bar(f2, x="urgency", y="tasa_SLA", title="Tasa de cumplimiento SLA por Urgency")
    fig3 = px.bar(f3, x="priority", y="tasa_SLA", title="Tasa de cumplimiento SLA por Priority")

    for fig in (fig1, fig2, fig3):
        fig.update_yaxes(tickformat=".0%", rangemode="tozero")

    return fig1, fig2, fig3

# =========================
# Estilos rápidos (CSS mínimo)
# =========================
APP_CSS = """
.container { max-width: 1100px; margin: 20px auto; font-family: system-ui, Arial; }
.row { display: flex; gap: 16px; flex-wrap: wrap; }
.col { flex: 1; min-width: 260px; }
.col-40 { flex: 0 0 40%; }
.col-60 { flex: 0 0 58%; }
.card { background: #fff; border: 1px solid #e5e7eb; border-radius: 8px; padding: 16px; box-shadow: 0 1px 2px rgba(0,0,0,.04); }
label { display:block; font-size: 12px; color:#555; margin-bottom: 6px; }
.btn { margin-top: 10px; padding: 8px 12px; border-radius: 8px; border: 1px solid #e5e7eb; background: #f3f4f6; cursor: pointer; }
"""

app.index_string = f"""
<!DOCTYPE html>
<html>
    <head>
        {{%metas%}}
        <title>{{%title%}}</title>
        {{%favicon%}}
        {{%css%}}
        <style>{APP_CSS}</style>
    </head>
    <body>
        {{%app_entry%}}
        <footer>
            {{%config%}}
            {{%scripts%}}
            {{%renderer%}}
        </footer>
    </body>
</html>
"""

if __name__ == "__main__":
    # Si instalas en servidor, cambia host/port según necesites
    app.run(debug=True)