import streamlit as st
import pandas as pd
import numpy as np

import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression

st.set_page_config(
    page_title="Ensayos Triaxiales",
    page_icon="../../utils/logo.png",
    initial_sidebar_state="collapsed",
)


def txcascadia(text, key="p", color="#2E3440"):
    text = '<{0} style="font-family:Cascadia Code;color:{1};">{2}</{3}>'.format(
        key, color, text, key
    )
    st.markdown(text, unsafe_allow_html=True)


@st.cache(allow_output_mutation=True)
def leer_registro(file):
    df = pd.read_csv(file, encoding="utf-8", names=["d (mm)", "ΔP (kN)", "ΔV (mm³)"])

    return df


###########
# Sidebar #
###########

with st.sidebar:
    txcascadia("Ensayos Triaxiales", "h1")

    txcascadia("Configuración ⚙️", "h2")

    unidad = st.select_slider("Unidades de Esfuerzo 📐", ("kPa", "kg/cm²"))

    st.write("")

    with open("example_data.zip", "rb") as example_data:
        st.download_button(
            label="¡Descargar data de ejemplo!",
            data=example_data,
            mime="application/zip",
        )


##############
# Encabezado #
##############

txcascadia("🔬 Ensayo Triaxial Consolidado Drenado", "h1")
st.markdown("---")

############################################
# Primer bloque: Datos de Espécimen Típico #
############################################

txcascadia("1️⃣ Datos de Espécimen Típico", "h2")

txcascadia("🏳️ Geometría | Remoldeo", "h3", "#434C5E")

col11, col12, col13 = st.columns([1.0, 1.0, 1.1])

with col11:
    h = st.number_input("Altura inicial (cm)", min_value=0.0, value=20.0, step=1.0)

with col12:
    D = st.number_input("Diámetro (cm)", min_value=0.0, value=10.0, step=1.0)

with col13:
    V = st.number_input(
        "Volumen inicial (cm³)",
        value=(3.1416 * D**2) / 4 * h,
        help="Volumen calculado a partir de las dimensiones iniciales del espécimen.",
        disabled=True,
    )

###################################################
# Segundo bloque: Lectura de Datos de Laboratorio #
###################################################

txcascadia("2️⃣ Registros de Laboratorio", "h2")

if "archivos" not in st.session_state:
    st.session_state["archivos"] = [None] * 3
if "ensayos" not in st.session_state:
    st.session_state["ensayos"] = [None] * 3
if "esf_conf" not in st.session_state:
    st.session_state["esf_conf"] = [None] * 3

h_cons = [None] * 3
v_cons = [None] * 3


txcascadia("✔️ Ensayo #1", "h3", "#434C5E")

with st.form("formensayo1"):

    verf_archivo1 = True

    col1, col2 = st.columns([1.0, 1.0])

    with col1:
        h_cons[0] = st.number_input(
            "Deformación vertical (mm)",
            min_value=0.0,
            key="h1",
            help="Deformación final producida en etapa de consolidación.",
            format="%.3f",
            step=0.001,
        )
        if not h_cons[0]:
            verf_archivo1 = False

    with col2:

        v_cons[0] = st.number_input(
            "Variación volumétrica (cm³)",
            min_value=0.0,
            key="v1",
            help="Volumen total desplazado en la etapa de consolidación.",
        )
        if not v_cons[0]:
            verf_archivo1 = False

    esf = st.number_input(
        "Esfuerzo de confinamiento ({})".format(unidad),
        step=100.0 if unidad == "kPa" else 1.0,
        min_value=0.0,
        key="esf1",
        help="Esfuerzo isotrópico final de la etapa de consolidación.",
    )
    if not esf:
        verf_archivo1 = False

    archivo1 = st.file_uploader(
        "Resultados del ensayo triaxial CD (.CSV)",
        type="csv",
        help="Registro de datos del equipo triaxial.",
        key="archivo1",
    )

    try:
        st.session_state.ensayos[0] = leer_registro(archivo1)
    except ValueError:
        verf_archivo1 = False

    col1, col2, col3 = st.columns(3)

    with col2:
        confirmar_ens1 = st.form_submit_button(label="🚀 Cargar 1° ensayo")


if not confirmar_ens1:
    pass
elif (
    confirmar_ens1 and verf_archivo1
):  # Agregar un checkbox para saber cuando ya este realizado
    st.session_state.archivos[0] = archivo1
    st.session_state.esf_conf[0] = esf
else:
    st.error("❌ Datos incompletos y/o registro inválido.")


txcascadia("✔️ Ensayo #2", "h3", "#434C5E")

with st.form("formensayo2"):

    verf_archivo2 = True

    col1, col2 = st.columns([1.0, 1.0])

    with col1:
        h_cons[1] = st.number_input(
            "Deformación vertical (mm)",
            min_value=0.0,
            key="h2",
            help="Deformación final producida en etapa de consolidación.",
            format="%.3f",
            step=0.001,
        )
        if not h_cons[1]:
            verf_archivo2 = False

    with col2:

        v_cons[1] = st.number_input(
            "Variación volumétrica (cm³)",
            min_value=0.0,
            key="v2",
            help="Volumen total desplazado en la etapa de consolidación.",
        )
        if not v_cons[1]:
            verf_archivo2 = False

    esf = st.number_input(
        "Esfuerzo de confinamiento ({})".format(unidad),
        step=100.0 if unidad == "kPa" else 1.0,
        min_value=0.0,
        key="esf2",
        help="Esfuerzo isotrópico final de la etapa de consolidación.",
    )
    if not esf:
        verf_archivo2 = False

    archivo2 = st.file_uploader(
        "Resultados del ensayo triaxial CD (.CSV)",
        type="csv",
        help="Registro de datos del equipo triaxial.",
        key="archivo2",
    )

    try:
        st.session_state.ensayos[1] = leer_registro(archivo2)
    except ValueError:
        verf_archivo2 = False

    col1, col2, col3 = st.columns(3)

    with col2:
        confirmar_ens2 = st.form_submit_button(label="🚀 Cargar 2° ensayo")


if not confirmar_ens2:
    pass
elif (
    confirmar_ens2 and verf_archivo2
):  # Agregar un checkbox para saber cuando ya este realizado
    st.session_state.archivos[1] = archivo2
    st.session_state.esf_conf[1] = esf
else:
    st.error("❌ Datos incompletos y/o registro inválido.")

txcascadia("✔️ Ensayo #3", "h3", "#434C5E")

with st.form("formensayo3"):

    verf_archivo3 = True

    col1, col2 = st.columns([1.0, 1.0])

    with col1:
        h_cons[2] = st.number_input(
            "Deformación vertical (mm)",
            min_value=0.0,
            key="h3",
            help="Deformación final producida en etapa de consolidación.",
            format="%.3f",
            step=0.001,
        )
        if not h_cons[2]:
            verf_archivo3 = False

    with col2:

        v_cons[2] = st.number_input(
            "Variación volumétrica (cm³)",
            min_value=0.0,
            key="v3",
            help="Volumen total desplazado en la etapa de consolidación.",
        )
        if not v_cons[2]:
            verf_archivo3 = False

    esf = st.number_input(
        "Esfuerzo de confinamiento ({})".format(unidad),
        step=100.0 if unidad == "kPa" else 1.0,
        min_value=0.0,
        key="esf3",
        help="Esfuerzo isotrópico final de la etapa de consolidación.",
    )
    if not esf:
        verf_archivo3 = False

    archivo3 = st.file_uploader(
        "Resultados del ensayo triaxial CD (.CSV)",
        type="csv",
        help="Registro de datos del equipo triaxial.",
        key="archivo3",
    )

    try:
        st.session_state.ensayos[2] = leer_registro(archivo3)
    except ValueError:
        verf_archivo3 = False

    col1, col2, col3 = st.columns(3)

    with col2:
        confirmar_ens3 = st.form_submit_button(label="🚀 Cargar 3° ensayo")


if not confirmar_ens3:
    pass
elif (
    confirmar_ens3 and verf_archivo3
):  # Agregar un checkbox para saber cuando ya este realizado
    st.session_state.archivos[2] = archivo3
    st.session_state.esf_conf[2] = esf
else:
    st.error("❌ Datos incompletos y/o registro inválido.")

if all(st.session_state.archivos):

    st.success("Todos los registros han sido cargados satisfactoriamente.")


############################################
# Tercer Bloque: Resultados de Laboratorio #
############################################

txcascadia("3️⃣ Resultados de Laboratorio", "h2")

# Visualización de Registros #
##############################

if all(st.session_state.archivos):
    txcascadia("📋 Visualización de Registros", "h3", "#434C5E")

    tabla_labo = st.selectbox(
        "Seleccione uno de los registros de laboratorio.",
        ("Ensayo #1", "Ensayo #2", "Ensayo #3"),
        key="visualizacion",
    )

    tablas = {
        "Ensayo #1": st.session_state.ensayos[0].iloc[:, [0, 1, 2]],
        "Ensayo #2": st.session_state.ensayos[1].iloc[:, [0, 1, 2]],
        "Ensayo #3": st.session_state.ensayos[2].iloc[:, [0, 1, 2]],
    }
    tabla = tablas[tabla_labo]

    fig = go.Figure(
        data=[
            go.Table(
                header=dict(
                    values=list(tabla.columns),
                    font_family="Cascadia Code",
                    font_size=13,
                    font_color="#ECEFF4",
                    fill_color="#5E81AC",
                    align="center",
                    height=26,
                ),
                cells=dict(
                    values=[tabla.iloc[:, 0], tabla.iloc[:, 1], tabla.iloc[:, 2]],
                    font_family="Cascadia Code",
                    font_size=13,
                    fill_color="#E5E9F0",
                    align="center",
                    height=24,
                    format=[".2f", ".2f", ".2f"],
                ),
            )
        ]
    )

    fig.update_layout(
        margin=dict(l=20, r=20, t=20, b=0),
        height=350,
    )

    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

# Gráficas de Laboratorio #
###########################

if all(st.session_state.archivos):
    txcascadia("📊 Gráficas de Laboratorio", "h3", "#434C5E")

    graflabo_escogido = st.selectbox(
        "Seleccione uno de los gráficos obtenidos en laboratorio.",
        (
            "Deformación Axial [d] - Fuerza Axial [ΔP]",
            "Deformación Axial [d] - Var. Volumétrica [ΔV]",
        ),
    )

    colores = ["#8FBCBB", "#88C0D0", "#81A1C1", "#5E81AC"]

    fig = go.Figure()

    if graflabo_escogido == "Deformación Axial [d] - Fuerza Axial [ΔP]":
        ptos_max = []

        for ensayo in st.session_state.ensayos:
            ptos_max.append(
                ensayo.loc[ensayo.iloc[:, 1] == max(ensayo.iloc[:, 1])].iloc[0]
            )

        for j, ensayo in enumerate(st.session_state.ensayos):
            fig.add_trace(
                go.Scatter(
                    x=ensayo["d (mm)"],
                    y=ensayo["ΔP (kN)"],
                    name="Ensayo #{}".format(j + 1),
                    line=dict(color=colores[j + 1]),
                    text="σ₀ = {}".format(st.session_state.esf_conf[j]),
                ),
            )

        for j, pair in enumerate(ptos_max):
            fig.add_trace(
                go.Scatter(
                    x=np.array(pair[0]),
                    y=np.array(pair[1]),
                    name="Ensayo #{}".format(j + 1),
                    marker_size=8,
                    marker_color=colores[j + 1],
                    text="σ₀ = {}".format(st.session_state.esf_conf[j]),
                ),
            )

            fig.add_annotation(
                x=pair[0],
                y=pair[1],
                text="ΔP = {:.2f}".format(pair[1]),
                showarrow=False,
                yshift=18,
                font={"color": colores[j + 1], "size": 13},
            )

        fig.update_xaxes(
            title_text="Deformación Axial, d (mm)",
            tickfont_size=13,
            fixedrange=True,
            showgrid=True,
            gridwidth=1,
            gridcolor="#D8DEE9",
            zeroline=False,
            mirror=True,
            showline=True,
            linecolor="#D8DEE9",
            linewidth=2,
        )

        fig.update_yaxes(
            title_text="Fuerza Axial, ΔP (kN)",
            tickfont_size=13,
            fixedrange=True,
            showgrid=True,
            gridwidth=1,
            gridcolor="#D8DEE9",
            zeroline=False,
            mirror=True,
            showline=True,
            linecolor="#D8DEE9",
            linewidth=2,
        )

    else:
        for j, ensayo in enumerate(st.session_state.ensayos):
            fig.add_trace(
                go.Scatter(
                    x=ensayo["d (mm)"],
                    y=ensayo["ΔV (mm³)"],
                    name="Ensayo #{}".format(j + 1),
                    line=dict(color=colores[j + 1]),
                    text="σ₀ = {}".format(st.session_state.esf_conf[j]),
                ),
            )

        fig.update_xaxes(
            title_text="Deformación Axial, d (mm)",
            tickfont_size=13,
            fixedrange=True,
            showgrid=True,
            gridwidth=1,
            gridcolor="#D8DEE9",
            zeroline=False,
            mirror=True,
            showline=True,
            linecolor="#D8DEE9",
            linewidth=2,
        )

        fig.update_yaxes(
            title_text="Variación Volumétrica, ΔV (mm³)",
            tickfont_size=13,
            fixedrange=True,
            showgrid=True,
            gridwidth=1,
            gridcolor="#D8DEE9",
            zeroline=False,
            mirror=True,
            showline=True,
            linecolor="#D8DEE9",
            linewidth=2,
        )

    fig.update_layout(
        dragmode="pan",
        font=dict(family="Cascadia Code", color="#5E81AC"),
        showlegend=False,
        template="seaborn",
        paper_bgcolor="#ECEFF4",
        plot_bgcolor="#E5E9F0",
        margin=dict(l=0, r=0, b=0, t=0),
        hoverlabel=dict(
            font_family="Cascadia Code",
        ),
    )

    st.plotly_chart(fig, config={"displayModeBar": False})
else:
    "⌛ Esperando resultados de laboratorio..."


############################################
# Cuarto bloque: Preprocesamiento de Datos #
############################################

txcascadia("4️⃣ Preprocesamiento de Datos", "h2")

if all(st.session_state.archivos):
    hf = [h - hc / 10 for hc in h_cons]
    hf_dict = dict(zip(["Ensayo #1", "Ensayo #2", "Ensayo #3"], hf))

    vf = [V - vc for vc in v_cons]
    vf_dict = dict(zip(["Ensayo #1", "Ensayo #2", "Ensayo #3"], vf))

    for i, ensayo in enumerate(st.session_state.ensayos):
        ensayo["ε (%)"] = ensayo["d (mm)"] / hf[i] * 10
        ensayo["ΔV (%)"] = (ensayo["ΔV (mm³)"] - ensayo["ΔV (mm³)"][0]) / vf[i] * 100
        ensayo["Ac (cm²)"] = (vf[i] - ensayo["ΔV (mm³)"]) / (
            hf[i] - ensayo["d (mm)"] / 10
        )
        ensayo["Δσ (kPa)"] = ensayo["ΔP (kN)"] / ensayo["Ac (cm²)"] * 10**4

    txcascadia("🏴 Geometría | Consolidación", "h3", "#434C5E")

    tabla_preprocs = st.selectbox(
        "Seleccione uno de los registros de laboratorio.",
        ("Ensayo #1", "Ensayo #2", "Ensayo #3"),
        key="preprocesamiento",
    )

    col41, col42 = st.columns(2)

    with col41:
        st.number_input(
            "Altura luego de la consolidación (cm)",
            format="%.3f",
            value=hf_dict[tabla_preprocs],
            key="hf",
            disabled=True,
        )

    with col42:
        st.number_input(
            "Volumen luego de la consolidación (cm³)",
            value=vf_dict[tabla_preprocs],
            key="vf",
            disabled=True,
        )

    txcascadia("📋 Resultados del Preprocesamiento", "h3", "#434C5E")

    tablas = {
        "Ensayo #1": st.session_state.ensayos[0].iloc[:, [3, 4, 5, 6]],
        "Ensayo #2": st.session_state.ensayos[1].iloc[:, [3, 4, 5, 6]],
        "Ensayo #3": st.session_state.ensayos[2].iloc[:, [3, 4, 5, 6]],
    }
    tabla = tablas[tabla_preprocs]

    fig = go.Figure(
        data=[
            go.Table(
                header=dict(
                    values=list(tabla.columns),
                    font_family="Cascadia Code",
                    font_size=13,
                    font_color="#ECEFF4",
                    fill_color="#4C566A",
                    align="center",
                    height=26,
                ),
                cells=dict(
                    values=[
                        tabla.iloc[:, 0],
                        tabla.iloc[:, 1],
                        tabla.iloc[:, 2],
                        tabla.iloc[:, 3],
                    ],
                    font_family="Cascadia Code",
                    font_size=13,
                    fill_color="#E5E9F0",
                    align="center",
                    height=24,
                    format=[".2f", ".3f", ".2f", ".2f"],
                ),
            )
        ]
    )

    fig.update_layout(
        margin=dict(l=20, r=20, t=20, b=0),
        height=350,
    )

    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    # Curvas del Preprocesamiento #
    ###############################

    txcascadia("📊 Curvas del Preprocesamiento", "h3", "#434C5E")

    grafpreprosc_escogido = st.selectbox(
        "Seleccione uno de los gráficos obtenidos del preprocesamiento.",
        (
            "Def. Axial Unitaria [ε] - Esfuerzo Desviador [Δσ]",
            "Def. Axial Unitaria [ε] - Def. Volumétrica Unitaria [ΔV]",
        ),
    )

    ptos_max = []

    for ensayo in st.session_state.ensayos:
        ptos_max.append(ensayo.loc[ensayo.iloc[:, 6] == max(ensayo.iloc[:, 6])].iloc[0])

    fig2 = go.Figure()

    if grafpreprosc_escogido == "Def. Axial Unitaria [ε] - Esfuerzo Desviador [Δσ]":

        for j, ensayo in enumerate(st.session_state.ensayos):
            fig2.add_trace(
                go.Scatter(
                    x=ensayo["ε (%)"],
                    y=ensayo["Δσ (kPa)"],
                    name="Ensayo #{}".format(j + 1),
                    line=dict(color=colores[j + 1]),
                    text="σ₀ = {}".format(st.session_state.esf_conf[j]),
                ),
            )

        for j, pair in enumerate(ptos_max):
            fig2.add_trace(
                go.Scatter(
                    x=np.array(pair[3]),
                    y=np.array(pair[6]),
                    name="Ensayo #{}".format(j + 1),
                    marker_size=8,
                    marker_color=colores[j + 1],
                    text="σ₀ = {}".format(st.session_state.esf_conf[j]),
                ),
            )

            fig2.add_annotation(
                x=pair[3],
                y=pair[6],
                text="Δσ = {:.2f}".format(pair[6]),
                showarrow=False,
                yshift=18,
                font={"color": colores[j + 1], "size": 13},
            )

        fig2.update_xaxes(
            title_text="Deformación Axial Unitaria, ε (%)",
            tickfont_size=13,
            fixedrange=True,
            showgrid=True,
            gridwidth=1,
            gridcolor="#D8DEE9",
            zeroline=False,
            mirror=True,
            showline=True,
            linecolor="#D8DEE9",
            linewidth=2,
        )

        fig2.update_yaxes(
            title_text="Esfuerzo Desviador, Δσ (kPa)",
            tickfont_size=13,
            fixedrange=True,
            showgrid=True,
            gridwidth=1,
            gridcolor="#D8DEE9",
            zeroline=False,
            mirror=True,
            showline=True,
            linecolor="#D8DEE9",
            linewidth=2,
        )

    else:
        for j, ensayo in enumerate(st.session_state.ensayos):
            fig2.add_trace(
                go.Scatter(
                    x=ensayo["ε (%)"],
                    y=ensayo["ΔV (%)"],
                    name="Ensayo #{}".format(j + 1),
                    line=dict(color=colores[j + 1]),
                    text="σ₀ = {}".format(st.session_state.esf_conf[j]),
                ),
            )

        fig2.update_xaxes(
            title_text="Deformación Axial Unitaria, ε (%)",
            tickfont_size=13,
            fixedrange=True,
            showgrid=True,
            gridwidth=1,
            gridcolor="#D8DEE9",
            zeroline=False,
            mirror=True,
            showline=True,
            linecolor="#D8DEE9",
            linewidth=2,
        )

        fig2.update_yaxes(
            title_text="Deformación Volumétrica Unitaria, ΔV (%)",
            tickfont_size=13,
            fixedrange=True,
            showgrid=True,
            gridwidth=1,
            gridcolor="#D8DEE9",
            zeroline=False,
            mirror=True,
            showline=True,
            linecolor="#D8DEE9",
            linewidth=2,
        )

    fig2.update_layout(
        dragmode="pan",
        font=dict(family="Cascadia Code", color="#5E81AC"),
        showlegend=False,
        template="seaborn",
        paper_bgcolor="#ECEFF4",
        plot_bgcolor="#E5E9F0",
        margin=dict(l=0, r=0, b=0, t=0),
        hoverlabel=dict(
            font_family="Cascadia Code",
        ),
    )

    st.plotly_chart(fig2, config={"displayModeBar": False})


else:
    "⌛ Esperando resultados de laboratorio..."

#############################################
# Quinto bloque: Procesamiento y Resultados #
#############################################

txcascadia("5️⃣ Procesamiento y Resultados", "h2")

if all(st.session_state.archivos):

    # Trayectorias de Esfuerzos #
    #############################

    txcascadia("📈 Trayectorias de Esfuerzos", "h3", "#434C5E")

    convencion = st.radio(
        "Escoga una convención para el cálculo de las invariantes de esfuerzos.",
        ("Massachusetts Institute of Technology", "University of Cambridge"),
    )

    if convencion == "University of Cambridge":
        for i, ensayo in enumerate(st.session_state.ensayos):
            ensayo["p' (kPa)"] = st.session_state.esf_conf[i] + ensayo["Δσ (kPa)"] / 3
            ensayo["q (kPa)"] = ensayo["Δσ (kPa)"]

            ensayo["p' [M.I.T]"] = st.session_state.esf_conf[i] + ensayo["Δσ (kPa)"] / 2
            ensayo["q [M.I.T]"] = ensayo["Δσ (kPa)"] / 2
    else:
        for i, ensayo in enumerate(st.session_state.ensayos):
            ensayo["p' (kPa)"] = st.session_state.esf_conf[i] + ensayo["Δσ (kPa)"] / 2
            ensayo["q (kPa)"] = ensayo["Δσ (kPa)"] / 2

    fig3 = go.Figure()

    for i, ensayo in enumerate(st.session_state.ensayos):
        fig3.add_trace(
            go.Scatter(
                x=ensayo["p' (kPa)"],
                y=ensayo["q (kPa)"],
                name="Ensayo #{}".format(i + 1),
                line=dict(color=colores[i + 1]),
                text="σ₀ = {}".format(st.session_state.esf_conf[i]),
            )
        )
        fig3.add_trace(
            go.Scatter(
                x=np.array(ensayo["p' (kPa)"].iloc[-1]),
                y=np.array(ensayo["q (kPa)"].iloc[-1]),
                name="Ensayo #{}".format(i + 1),
                marker_size=7,
                marker_color=colores[i + 1],
                text="σ₀ = {}".format(st.session_state.esf_conf[i]),
            )
        )

    if convencion == "University of Cambridge":
        p_falla_MIT = np.array(
            [ensayo["p' [M.I.T]"].iloc[-1] for ensayo in st.session_state.ensayos]
        ).reshape(-1, 1)
        q_falla_MIT = np.array(
            [ensayo["q [M.I.T]"].iloc[-1] for ensayo in st.session_state.ensayos]
        )

        modelo_MIT = LinearRegression()
        modelo_MIT.fit(p_falla_MIT, q_falla_MIT)

    p_falla = np.array(
        [ensayo["p' (kPa)"].iloc[-1] for ensayo in st.session_state.ensayos]
    ).reshape(-1, 1)
    q_falla = np.array(
        [ensayo["q (kPa)"].iloc[-1] for ensayo in st.session_state.ensayos]
    )

    modelo = LinearRegression()
    modelo.fit(p_falla, q_falla)

    alpha = float(np.degrees(np.arctan(modelo.coef_)))
    k = modelo.intercept_

    rango_p = np.linspace(0, p_falla.max(), 100)
    rango_q = modelo.predict(rango_p.reshape(-1, 1))

    fig3.add_trace(
        go.Scatter(
            x=rango_p,
            y=rango_q,
            name="Envolvente M.",
            mode="lines",
            line={"dash": "dash", "color": "#4C566A", "width": 2},
            opacity=0.5,
        )
    )

    fig3.add_annotation(
        x=rango_p[65],
        y=rango_q[65],
        text="α = {:.2f}° ‖ k = {:.2f}".format(alpha, k),
        showarrow=True,
        arrowhead=0,
        arrowwidth=2,
        arrowcolor="#4C566A",
        bordercolor="#4C566A",
        borderwidth=1.5,
        borderpad=7,
        bgcolor="#ECEFF4",
        font={"color": "#434C5E", "size": 12.5},
    )

    fig3.update_xaxes(
        title_text="p' (kPa)",
        tickfont_size=13,
        fixedrange=True,
        showgrid=True,
        gridwidth=1,
        gridcolor="#D8DEE9",
        zeroline=False,
        mirror=True,
        showline=True,
        linecolor="#D8DEE9",
        linewidth=2,
    )

    fig3.update_yaxes(
        title_text="q (kPa)",
        tickfont_size=13,
        fixedrange=True,
        showgrid=True,
        gridwidth=1,
        gridcolor="#D8DEE9",
        zeroline=False,
        mirror=True,
        showline=True,
        linecolor="#D8DEE9",
        linewidth=2,
    )

    fig3.update_layout(
        dragmode="pan",
        font=dict(family="Cascadia Code", color="#5E81AC"),
        showlegend=False,
        template="seaborn",
        paper_bgcolor="#ECEFF4",
        plot_bgcolor="#E5E9F0",
        margin=dict(l=0, r=0, b=0, t=0),
        hoverlabel=dict(font_family="Cascadia Code"),
    )

    st.write("")
    st.plotly_chart(fig3, config={"displayModeBar": False})

    # Círculos de Mohr #
    ####################

    txcascadia("📉 Círculos de Mohr", "h3", "#434C5E")

    if convencion == "University of Cambridge":
        phi = float(np.degrees(np.arcsin(modelo_MIT.coef_)))
        c = modelo_MIT.intercept_ / np.cos(np.deg2rad(phi))
    else:
        phi = float(np.degrees(np.arcsin(modelo.coef_)))
        c = modelo.intercept_ / np.cos(np.deg2rad(phi))

    fig4 = go.Figure()

    for j, pair in enumerate(ptos_max):
        fig4.add_shape(
            type="circle",
            xref="x",
            yref="y",
            name="xd",
            x0=st.session_state.esf_conf[j],
            y0=-pair[6] / 2,
            x1=st.session_state.esf_conf[j] + pair[6],
            y1=pair[6] / 2,
            opacity=1.0,
            line=dict(color=colores[j + 1], width=2),
            fillcolor="rgba(0,0,0,0)",
        )

    rango_p = [0, max(st.session_state.esf_conf) + max(pair[6] for pair in ptos_max)]
    rango_q1 = [np.tan(np.deg2rad(phi)) * p + c for p in rango_p]
    rango_q2 = [-np.tan(np.deg2rad(phi)) * p - c for p in rango_p]

    fig4.add_trace(
        go.Scatter(
            x=rango_p,
            y=rango_q1,
            mode="lines",
            line={"dash": "dash", "color": "#4C566A", "width": 2},
            opacity=0.5,
        )
    )

    fig4.add_trace(
        go.Scatter(
            x=rango_p,
            y=rango_q2,
            mode="lines",
            line={"dash": "dash", "color": "#4C566A", "width": 2},
            opacity=0.5,
        )
    )

    fig4.update_xaxes(
        title_text="σ' (kPa)",
        tickfont_size=13,
        showgrid=True,
        gridwidth=1,
        gridcolor="#D8DEE9",
        zeroline=False,
        mirror=True,
        showline=True,
        linecolor="#D8DEE9",
        linewidth=2,
    )

    fig4.update_yaxes(
        range=rango_q1,
        title_text="τ (kPa)",
        tickfont_size=13,
        showgrid=True,
        gridwidth=1,
        gridcolor="#D8DEE9",
        scaleanchor="x",
        scaleratio=1,
        zeroline=False,
        mirror=True,
        showline=True,
        linecolor="#D8DEE9",
        linewidth=2,
    )

    fig4.add_annotation(
        x=2 * rango_p[1] / 3,
        y=2 * rango_q1[1] / 3,
        text="Φ = {:.2f}° ‖ c = {:.2f}".format(phi, c),
        showarrow=True,
        arrowhead=0,
        arrowwidth=2,
        arrowcolor="#4C566A",
        bordercolor="#4C566A",
        borderwidth=1.5,
        borderpad=7,
        bgcolor="#ECEFF4",
        font={"color": "#434C5E", "size": 12.5},
    )

    fig4.update_layout(
        dragmode="pan",
        font=dict(family="Cascadia Code", color="#5E81AC"),
        showlegend=False,
        template="seaborn",
        paper_bgcolor="#ECEFF4",
        plot_bgcolor="#E5E9F0",
        margin=dict(l=0, r=0, b=0, t=0),
        hoverlabel=dict(font_family="Cascadia Code"),
    )

    st.write("")
    st.plotly_chart(fig4, config={"displayModeBar": False})

    # Resumen de Resultados #
    ########################

    txcascadia("🎈 Resumen de Resultados", "h3", "#434C5E")

    resultados = pd.DataFrame(
        {
            "Ensayos": ["Ensayo #1", "Ensayo #2", "Ensayo #3"],
            "σ3 ({0})".format(unidad): [
                st.session_state["esf_conf"][0],
                st.session_state["esf_conf"][1],
                st.session_state["esf_conf"][2],
            ],
            "Δσf ({0})".format(unidad): [
                max(st.session_state.ensayos[0]["Δσ (kPa)"]),
                max(st.session_state.ensayos[1]["Δσ (kPa)"]),
                max(st.session_state.ensayos[2]["Δσ (kPa)"]),
            ],
        }
    )

    fig5 = go.Figure(
        data=[
            go.Table(
                header=dict(
                    values=list(resultados.columns),
                    font_family="Cascadia Code",
                    font_size=13,
                    font_color="#ECEFF4",
                    fill_color="#4C566A",
                    align="center",
                    height=30,
                ),
                cells=dict(
                    values=[
                        resultados.iloc[:, 0],
                        resultados.iloc[:, 1],
                        resultados.iloc[:, 2],
                    ],
                    font_family="Cascadia Code",
                    font_size=13,
                    fill_color="#E5E9F0",
                    align="center",
                    height=26,
                    format=["", ".2f", ".2f"],
                ),
            )
        ]
    )

    fig5.update_layout(
        margin=dict(l=20, r=20, t=20, b=0),
        height=350,
    )

    st.plotly_chart(fig5, use_container_width=True, config={"displayModeBar": False})

else:
    st.write("⌛ Esperando resultados de laboratorio...")
