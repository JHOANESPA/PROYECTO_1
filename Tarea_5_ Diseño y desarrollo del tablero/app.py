import pandas as pd
import numpy as np
from pathlib import Path
import plotly.express as px
from dash import Dash, dcc, html, Input, Output, State
import tensorflow as tf
import keras
from sklearn.preprocessing import StandardScaler

# =========================
# Rutas y artefactos
# =========================
HERE = Path(__file__).resolve().parent
MODEL_PATH = HERE / "modelo3_sla_tf_stdnum.h5"
COLS_PATH  = HERE / "columnas_modelo3.csv"
DATA_PATH  = HERE / "incident_event_log.csv"

# Lee columnas y elimina índice fantasma "0" si existe
cols_raw = pd.read_csv(COLS_PATH, header=None).iloc[:, 0]
feature_columns = cols_raw[cols_raw != "0"].tolist()

# Modelo
model = keras.models.load_model(MODEL_PATH)

# =========================
# Datos base y scaler
# =========================
df = pd.read_csv(DATA_PATH)
df3= df.copy()
df.replace('?', pd.NA, inplace=True)
df = df.drop_duplicates()
df.fillna({'resolved_at': df['closed_at']}, inplace=True)

for c in ["opened_at", "resolved_at", "closed_at"]:
    df[c] = pd.to_datetime(df[c], format="%d/%m/%Y %H:%M", errors="coerce")

# Un registro por incidente y variable de tiempo (días)
df = df.drop_duplicates(subset="number", keep="last").copy()
df["resolution_time"] = (df["resolved_at"] - df["opened_at"]).dt.total_seconds() / 3600 / 24

# Scaler SOLO en numéricas del dataset original
scaler = StandardScaler()
scaler.fit(df[["reassignment_count", "reopen_count", "sys_mod_count"]])

# =========================
# Diccionario (para tooltips ℹ️)
# =========================
diccionario = {
    "reassignment_count": "Número de veces que el incidente cambió de grupo o analista",
    "reopen_count": "Número de veces que el usuario rechazó la resolución",
    "sys_mod_count": "Cantidad de actualizaciones hechas al incidente",
    "impact": "Nivel de impacto causado (1: Alto, 2: Medio, 3: Bajo)",
    "urgency": "Urgencia indicada por el usuario (1: Alta, 2: Media, 3: Baja)",
    "priority": "Prioridad calculada (1: Crítica, 2: Alta, 3: Media, 4: Baja)",
    "category": "Número de la categoría del servicio afectado",
    "assignment_group": "Número del grupo de soporte asignado",
    "knowledge": "Indica si se usó la base de conocimiento"
}

# Mapas de etiquetas (visual) y órdenes fijos en español
URG_MAP  = {"1 - High":"Alta", "2 - Medium":"Media", "3 - Low":"Baja"}
PRIO_MAP = {"1 - Critical":"Crítica", "2 - High":"Alta", "3 - Moderate":"Media", "4 - Low":"Baja"}
ORDER_URG  = ["Alta", "Media", "Baja"]
ORDER_PRIO = ["Crítica", "Alta", "Media", "Baja"]

# =========================
# Utilidades
# =========================
def preprocess_single_row(row_dict: dict, feature_columns: list) -> pd.DataFrame:
    df_row = pd.DataFrame([row_dict])

    # Numéricas
    for c in ["reassignment_count", "reopen_count", "sys_mod_count"]:
        df_row[c] = pd.to_numeric(df_row[c], errors="coerce").fillna(0)

    # Escalar SOLO las numéricas
    df_row[["reassignment_count", "reopen_count", "sys_mod_count"]] = scaler.transform(
        df_row[["reassignment_count", "reopen_count", "sys_mod_count"]]
    )

    # Dummies de categóricas
    X_cat = pd.get_dummies(
        df_row[["impact", "urgency", "priority", "category", "assignment_group", "knowledge"]],
        drop_first=True
    )
    X_num = df_row[["reassignment_count", "reopen_count", "sys_mod_count"]]

    # Unir y ALINEAR exactamente con columnas del entrenamiento
    X = pd.concat([X_num, X_cat], axis=1).astype("float32")
    X = X.reindex(columns=feature_columns, fill_value=0.0).astype("float32")
    return X

def exists_category_group(cat, grp):
    cat_full = f"Category {cat}"
    grp_full = f"Group {grp}"
    cats = df['category'].dropna().unique()
    grps = df['assignment_group'].dropna().unique()
    return cat_full in cats, grp_full in grps

# =========================
# App
# =========================
app = Dash(__name__)
app.title = "Proyecto 1"

def campo(label_es, id_, type_="text", value=None, info=""):
    """Campo con tooltip ℹ️ (solo pasar el cursor)."""
    return html.Div([
        html.Label([
            label_es,
            html.Span(
                " ℹ️",
                title=f"{diccionario[info]} (Nombre en los datos: {info}). "
                      ,
                style={"cursor":"help","fontSize":"14px","marginLeft":"4px"}
            )
        ], style={"display":"block","fontWeight":"600","marginBottom":"4px"}),
        dcc.Input(
            id=id_, type=type_, value=value,
            style={"width":"100%","padding":"6px","borderRadius":"6px","border":"1px solid #ccc"}
        )
    ], className="campo")

# -------------------------
# Layout: Predicción
# -------------------------
form_layout = html.Div([
    html.P("Nota: para conocer la descripción de cada parámetro, "
           "pasa el cursor sobre el ícono ℹ️ junto al nombre (no hagas clic).",
           style={"fontSize":"14px","color":"#333","margin":"0 0 12px"}),

    html.Div([
        html.Div(campo("Conteo de reasignaciones","inp-reassign","number",3,"reassignment_count"), className="col"),
        html.Div(campo("Conteo de reaperturas","inp-reopen","number",0,"reopen_count"), className="col"),
        html.Div(campo("Cantidad de modificaciones","inp-sysmod","number",10,"sys_mod_count"), className="col"),
    ], className="row"),

    html.Div([
        html.Div([
            html.Label(["Impacto",
                html.Span(" ℹ️", title=f"{diccionario['impact']} (Nombre en los datos: impact). "
                                        "Solo pasa el cursor.", style={"cursor":"help","marginLeft":"4px"})
            ]),
            dcc.Dropdown(["1 - High","2 - Medium","3 - Low"], "1 - High", id="inp-impact", clearable=False)
        ], className="col"),
        html.Div([
            html.Label(["Urgencia",
                html.Span(" ℹ️", title=f"{diccionario['urgency']} (Nombre en los datos: urgency). "
                                        "Solo pasa el cursor.", style={"cursor":"help","marginLeft":"4px"})
            ]),
            dcc.Dropdown(["1 - High","2 - Medium","3 - Low"], "1 - High", id="inp-urgency", clearable=False)
        ], className="col"),
        html.Div([
            html.Label(["Prioridad",
                html.Span(" ℹ️", title=f"{diccionario['priority']} (Nombre en los datos: priority). "
                                        "Solo pasa el cursor.", style={"cursor":"help","marginLeft":"4px"})
            ]),
            dcc.Dropdown(["1 - Critical","2 - High","3 - Moderate","4 - Low"], "1 - Critical", id="inp-priority", clearable=False)
        ], className="col"),
    ], className="row"),

    html.Div([
        html.Div(campo("Categoría del servicio","inp-category","number",40,"category"), className="col"),
        html.Div(campo("Grupo asignado","inp-group","number",56,"assignment_group"), className="col"),
        html.Div([
            html.Label(["Uso de base de conocimiento",
                html.Span(" ℹ️", title=f"{diccionario['knowledge']} (Nombre en los datos: knowledge). "
                                        "Solo pasa el cursor.", style={"cursor":"help","marginLeft":"4px"})
            ]),
            dcc.Dropdown(
                options=[{"label":"Sí","value":True},{"label":"No","value":False}],
                value=True, id="inp-knowledge", clearable=False
            )
        ], className="col"),
    ], className="row"),

    html.Button("Calcular probabilidad", id="btn-predict", className="boton"),
    html.Div(id="pred-output", className="pred-box")  # cuadro grande, centrado
], className="form-card")

# -------------------------
# Layout: Exploración
# -------------------------
eda_layout = html.Div([
    html.Div([
        html.Div(dcc.Graph(id="fig-sla-pie"), className="col"),
        html.Div(dcc.Graph(id="fig-resolucion"), className="col"),
    ], className="row"),

    html.Div([
        html.Div(dcc.Graph(id="fig-sla_urg_prio"), className="col"),
        html.Div(dcc.Graph(id="fig-tiempo_urg_prio"), className="col"),
    ], className="row"),

    html.Div([
        html.Div(dcc.Graph(id="fig-knowledge"), className="col"),
        html.Div(dcc.Graph(id="fig-scatter"), className="col"),
    ], className="row"),
])

# -------------------------
# Layout principal
# -------------------------
app.layout = html.Div([
    html.Div([
        html.Div("📊", style={"fontSize":"40px","marginBottom":"8px"}),
        html.H1("Proyecto 1: Analítica computacional para la toma de decisiones"),
        html.H3("Dashboard de predicción de cumplimiento de SLA", style={"color":"#444"}),
    ], style={"textAlign":"center","marginBottom":"30px"}),

    html.P(
        "Este dashboard permite analizar el comportamiento de los incidentes registrados, "
        "explorar patrones de resolución y predecir la probabilidad de romper el SLA para un nuevo incidente, "
        "considerando sus características principales.",
        style={"maxWidth":"900px","margin":"0 auto 40px","textAlign":"center","fontSize":"16px"}
    ),

    dcc.Tabs(id="tabs", value="tab-2", children=[
        dcc.Tab(label="📊 Exploración de Datos", value="tab-2", children=[eda_layout]),
        dcc.Tab(label="🔮 Predicción", value="tab-1", children=[form_layout]),
    ])
], className="container")

# =========================
# Callbacks
# =========================
@app.callback(
    Output("pred-output","children"),
    Input("btn-predict","n_clicks"),
    State("inp-reassign","value"),
    State("inp-reopen","value"),
    State("inp-sysmod","value"),
    State("inp-impact","value"),
    State("inp-urgency","value"),
    State("inp-priority","value"),
    State("inp-category","value"),
    State("inp-group","value"),
    State("inp-knowledge","value"),
    prevent_initial_call=True
)
def make_prediction(nc, reassignment, reopen, sysmod, impact, urgency, priority, category, group, knowledge):
    cat_ok, grp_ok = exists_category_group(category, group)
    if not cat_ok or not grp_ok:
        return html.Div(
            f"⚠️ Categoría {'OK' if cat_ok else 'NO encontrada'} | Grupo {'OK' if grp_ok else 'NO encontrado'}",
            style={"color":"#b42318","textAlign":"center","fontSize":"24px","padding":"20px","border":"1px solid #f0c2c2","borderRadius":"10px","background":"#fff5f5"}
        )

    row = {
        "reassignment_count": reassignment,
        "reopen_count": reopen,
        "sys_mod_count": sysmod,
        "impact": impact,
        "urgency": urgency,
        "priority": priority,
        "category": f"Category {category}",
        "assignment_group": f"Group {group}",
        "knowledge": knowledge,
    }

    X = preprocess_single_row(row, feature_columns)
    prob = float(model.predict(X, verbose=0)[0][0])
    pct  = f"{prob*100:.2f}%"

    if prob < 0.33:
        color, txt = "#1a7f37", "Riesgo BAJO de romper SLA"
        bg = "#eaf6ed"
    elif prob < 0.66:
        color, txt = "#b7791f", "Riesgo MEDIO de romper SLA"
        bg = "#fff6e5"
    else:
        color, txt = "#b42318", "Riesgo ALTO de romper SLA"
        bg = "#ffecec"

    return html.Div(
        f"Probabilidad: {pct} – {txt}",
        style={"color":color,"textAlign":"center","fontSize":"26px","padding":"20px",
               "border":"1px solid #e5e7eb","borderRadius":"10px","background":bg}
    )

@app.callback(
    Output("fig-sla-pie","figure"),
    Output("fig-resolucion","figure"),
    Output("fig-sla_urg_prio","figure"),
    Output("fig-tiempo_urg_prio","figure"),
    Output("fig-knowledge","figure"),
    Output("fig-scatter","figure"),
    Input("tabs","value")
)
def make_eda_graphs(tab):
    if tab != "tab-2":
        return [px.scatter(title="")]*6

    df_plot = df.copy()
    df_plot["Cumplió SLA"] = df_plot["made_sla"].map({True:"Sí", False:"No"})

    # 1) Pie SLA
    fig1 = px.pie(
        values=df_plot['Cumplió SLA'].value_counts().values,
        names=df_plot['Cumplió SLA'].value_counts().index,
        title="<b>Proporción de incidentes que cumplen o rompen el SLA</b>"
    )

    # 2) Histograma (log Y)
    fig2 = px.histogram(
        df_plot, x="resolution_time", color="Cumplió SLA",
        nbins=50, log_y=True,
        title="<b>Distribución del tiempo de resolución según cumplimiento del SLA</b>"
    )
    fig2.update_xaxes(title="<b>Tiempo de resolución (días)</b>")
    fig2.update_yaxes(title="<b>Cantidad de incidentes</b>")

    # 3) Heatmap % SLA 
    #    No se cambian los valores; solo se cambian los textos de ticks en los ejes.
    sla_group_cat = df3.pivot_table(
        index="urgency",
        columns="priority",
        values="made_sla",
        aggfunc="mean"
    )
    fig3 = px.imshow(
        sla_group_cat, text_auto=".2f", color_continuous_scale="RdYlGn",
        title="<b>% de SLA cumplido por urgencia y prioridad</b>"
    )
    # Etiquetas de ejes en español (manteniendo el orden/calculo original)
    fig3.update_xaxes(
        title="<b>Prioridad del incidente</b>",
        ticktext=[PRIO_MAP.get(c, c) for c in sla_group_cat.columns],
        tickvals=list(range(len(sla_group_cat.columns)))
    )
    fig3.update_yaxes(
        title="<b>Urgencia informada por cliente</b>",
        ticktext=[URG_MAP.get(r, r) for r in sla_group_cat.index],
        tickvals=list(range(len(sla_group_cat.index)))
    )

    # 4) Heatmap tiempo promedio
    sla_time = df_plot.pivot_table(index="urgency", columns="priority", values="resolution_time", aggfunc="mean")
    fig4 = px.imshow(
        sla_time, text_auto=".2f", color_continuous_scale="RdYlGn_r",
        title="<b>Tiempo promedio de resolución (días) por Urgencia y Prioridad</b>"
    )
    fig4.update_xaxes(
        title="<b>Prioridad del incidente</b>",
        ticktext=[PRIO_MAP.get(c, c) for c in sla_time.columns],
        tickvals=list(range(len(sla_time.columns)))
    )
    fig4.update_yaxes(
        title="<b>Urgencia informada por cliente</b>",
        ticktext=[URG_MAP.get(r, r) for r in sla_time.index],
        tickvals=list(range(len(sla_time.index)))
    )

    # 5) Pie knowledge
    tabla = pd.crosstab(df_plot['knowledge'], df_plot['Cumplió SLA'])
    fig5 = px.pie(
        values=tabla.sum(axis=1),
        names=["No","Sí"],
        title="<b>Uso de base de conocimiento en incidentes</b>"
    )

    # 6) Dispersión
    fig6 = px.scatter(
        df_plot, x="resolution_time", y="sys_mod_count", color="Cumplió SLA",
        title="<b>Relación entre tiempo de resolución y cantidad de modificaciones</b>"
    )
    fig6.update_xaxes(title="<b>Tiempo de resolución (días)</b>")
    fig6.update_yaxes(title="<b>Cantidad de modificaciones</b>")

    # Títulos centrados y en negro
    for fig in [fig1, fig2, fig3, fig4, fig5, fig6]:
        fig.update_layout(
            title={'x':0.5,'xanchor':'center'},
            title_font=dict(size=16, color="#000", family="Segoe UI"),
            margin={'t':90}
        )

    return fig1, fig2, fig3, fig4, fig5, fig6

# =========================
# CSS incrustado
# =========================
app.index_string = """
<!DOCTYPE html>
<html>
<head>
    {%metas%}
    <title>{%title%}</title>
    {%favicon%}
    {%css%}
    <style>
        body { font-family: 'Segoe UI', system-ui, Arial; background: #f7f8fa; }
        .container { max-width: 1120px; margin: 22px auto; }
        .row { display: flex; gap: 16px; flex-wrap: wrap; margin-bottom: 24px; }
        .col { flex: 1; min-width: 260px; background:#fff; border:1px solid #eee; border-radius:10px; padding:16px; }
        .form-card { background:#fff; padding:20px; border-radius:10px; box-shadow:0 2px 4px rgba(0,0,0,.05); }
        .campo { margin-bottom: 16px; }
        .boton { margin-top: 10px; padding: 10px 16px; border:none; background:#0069d9; color:#fff; border-radius:8px; cursor:pointer; display:block; margin-left:auto; margin-right:auto; }
        .boton:hover { background:#0053b3; }
        .pred-box { margin-top:18px; text-align:center; font-size:24px; }
        label { font-weight: 600; }
    </style>
</head>
<body>
    {%app_entry%}
    <footer>
        {%config%}
        {%scripts%}
        {%renderer%}
    </footer>
</body>
</html>
"""

if __name__ == "__main__":
    app.run(debug=True)

