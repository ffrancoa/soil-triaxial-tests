import os
import sys

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

from sklearn.linear_model import LinearRegression


ROOT_DIR = os.path.realpath(os.path.join(os.path.dirname(__file__), "..", ".."))


st.set_page_config(
    page_title="Soil Triaxial Tests",
    page_icon=os.path.join(ROOT_DIR, "media", "logo.png"),
    initial_sidebar_state="collapsed",
)

sys.path.append(os.path.join(ROOT_DIR, "utils"))

import st_utils as _st

st.markdown(_st.CUSTOM_FONT_URL, unsafe_allow_html=True)


@st.cache(allow_output_mutation=True)
def _read_csv(file):
    df = pd.read_csv(file, encoding="utf-8", names=["d (mm)", "ΔP", "Vb (cm³)"])

    return df


###########
# Sidebar #
###########

with st.sidebar:
    _st.googlef_text("Soil Triaxial Tests", key="h1")

    _st.googlef_text("Settings ⚙️", key="h2")

    unit = st.selectbox("Stress Units 📐", ("kPa", "kg/cm²"))

    if unit == "kPa":
        unit2 = "kN"
    else:
        unit2 = "kg"

    st.write("")

    examples_path = os.path.join(ROOT_DIR, "cd-test", "data", "example_data_cd.zip")

    with open(examples_path, "rb") as example_data:
        st.download_button(
            label="📄 Click here to download example data!",
            data=example_data,
            file_name="example_data.zip",
            mime="application/zip",
        )


##############
# Encabezado #
##############

_st.googlef_text("Consolidated Drained Triaxial Test", key="h1")

############################################
# Primer bloque: Datos de Espécimen Típico #
############################################

_st.googlef_text("1️⃣ Tipical Specimen Parameters", key="h2")

_st.googlef_text("🏳️ Geometry | Preparation", key="h3", color="#434C5E")

col11, col12, col13 = st.columns([1.0, 1.0, 1.1])

with col11:
    h = st.number_input("Initial height (cm)", min_value=0.0, value=20.0, step=1.0)

with col12:
    D = st.number_input("Diameter (cm)", min_value=0.0, value=10.0, step=1.0)

with col13:
    V = st.number_input(
        "Initial volume (cm³)",
        value=(3.1416 * D**2) / 4 * h,
        help="Volume calculated since the initial conditions of the specimen.",
        disabled=True,
    )

###################################################
# Segundo bloque: Lectura de Datos de Laboratorio #
###################################################

_st.googlef_text("2️⃣ Laboratory Records", key="h2")

if "files" not in st.session_state:
    st.session_state["files"] = [None] * 3
if "tests" not in st.session_state:
    st.session_state["tests"] = [None] * 3
if "esf_conf" not in st.session_state:
    st.session_state["esf_conf"] = [None] * 3

h_cons = [None] * 3
v_cons = [None] * 3


_st.googlef_text("✔️ Test #1", key="h3", color="#434C5E")

with st.form("formensayo1"):

    file1_check = True

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

    with col2:

        v_cons[0] = st.number_input(
            "Variación volumétrica (cm³)",
            min_value=0.0,
            key="v1",
            help="Volumen total desplazado en la etapa de consolidación.",
        )

    esf = st.number_input(
        "Esfuerzo de confinamiento ({})".format(unit),
        step=100.0 if unit == "kPa" else 1.0,
        min_value=0.0,
        key="esf1",
        help="Esfuerzo isotrópico final de la etapa de consolidación.",
    )
    if esf in st.session_state["esf_conf"]:
        file1_check = False
        error_message = "Radial stresses must be different from each other."
    elif not esf:
        file1_check = False
        error_message = "Radial stress must be indicated."

    file1 = st.file_uploader(
        "Resultados del ensayo triaxial CD (.CSV)",
        type="csv",
        help="Registro de datos del equipo triaxial.",
        key="file1",
    )

    try:
        st.session_state.tests[0] = _read_csv(file1)
    except ValueError:
        file1_check = False
        error_message = "Invalid test record."

    col1, col2, col3 = st.columns(3)

    with col2:
        test1_verf = st.form_submit_button(label="🚀 Load 1st test!")


if not test1_verf:
    pass
elif (
    test1_verf and file1_check
):  # Agregar un checkbox para saber cuando ya este realizado
    st.session_state.files[0] = file1
    st.session_state.esf_conf[0] = esf
else:
    st.error("❌ {0}".format(error_message))


_st.googlef_text("✔️ Test #2", key="h3", color="#434C5E")

with st.form("formensayo2"):

    file2_check = True

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

    with col2:

        v_cons[1] = st.number_input(
            "Variación volumétrica (cm³)",
            min_value=0.0,
            key="v2",
            help="Volumen total desplazado en la etapa de consolidación.",
        )
    esf = st.number_input(
        "Esfuerzo de confinamiento ({})".format(unit),
        step=100.0 if unit == "kPa" else 1.0,
        min_value=0.0,
        key="esf2",
        help="Esfuerzo isotrópico final de la etapa de consolidación.",
    )
    if esf in st.session_state["esf_conf"]:
        file2_check = False
        error_message = "Radial stresses must be different from each other."
    elif not esf:
        file2_check = False
        error_message = "Radial stress must be indicated."

    file2 = st.file_uploader(
        "Resultados del ensayo triaxial CD (.CSV)",
        type="csv",
        help="Registro de datos del equipo triaxial.",
        key="file2",
    )

    try:
        st.session_state.tests[1] = _read_csv(file2)
    except ValueError:
        file2_check = False
        error_message = "Invalid test record."

    col1, col2, col3 = st.columns(3)

    with col2:
        test2_verf = st.form_submit_button(label="🚀 Load 2nd test!")


if not test2_verf:
    pass
elif (
    test2_verf and file2_check
):  # TODO: Agregar un checkbox para saber cuando ya este realizado
    st.session_state.files[1] = file2
    st.session_state.esf_conf[1] = esf
else:
    st.error("❌ {0}".format(error_message))

_st.googlef_text("✔️ Test #3", key="h3", color="#434C5E")

with st.form("formensayo3"):

    file3_check = True

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

    with col2:

        v_cons[2] = st.number_input(
            "Variación volumétrica (cm³)",
            min_value=0.0,
            key="v3",
            help="Volumen total desplazado en la etapa de consolidación.",
        )

    esf = st.number_input(
        "Esfuerzo de confinamiento ({})".format(unit),
        step=100.0 if unit == "kPa" else 1.0,
        min_value=0.0,
        key="esf3",
        help="Esfuerzo isotrópico final de la etapa de consolidación.",
    )
    if esf in st.session_state["esf_conf"]:
        file3_check = False
        error_message = "Radial stresses must be different from each other."
    elif not esf:
        file3_check = False
        error_message = "Radial stress must be indicated."

    file3 = st.file_uploader(
        "Resultados del ensayo triaxial CD (.CSV)",
        type="csv",
        help="Registro de datos del equipo triaxial.",
        key="file3",
    )

    try:
        st.session_state.tests[2] = _read_csv(file3)
    except ValueError:
        file3_check = False
        error_message = "Invalid test record."

    col1, col2, col3 = st.columns(3)

    with col2:
        test3_verf = st.form_submit_button(label="🚀 Load 3rd test!")


if not test3_verf:
    pass
elif (
    test3_verf and file3_check
):  # Agregar un checkbox para saber cuando ya este realizado
    st.session_state.files[2] = file3
    st.session_state.esf_conf[2] = esf
else:
    st.error("❌ {0}".format(error_message))

if all(st.session_state.files):

    st.success("Todos los registros han sido cargados satisfactoriamente.")


############################################
# Tercer Bloque: Resultados de Laboratorio #
############################################

_st.googlef_text("3️⃣ Laboratory Results", key="h2")

# Visualización de Registros #
##############################

if all(st.session_state.files):
    _st.googlef_text("📋 Records Tables", key="h3", color="#434C5E")

    lab_table = st.selectbox(
        "Seleccione uno de los registros de laboratorio.",
        ("Test #1", "Test #2", "Test #3"),
        key="visualizacion",
    )

    tables = {
        "Test #1": st.session_state.tests[0].iloc[:, [0, 1, 2]],
        "Test #2": st.session_state.tests[1].iloc[:, [0, 1, 2]],
        "Test #3": st.session_state.tests[2].iloc[:, [0, 1, 2]],
    }
    tabla = tables[lab_table]

    fig = go.Figure(
        data=[
            go.Table(
                header=dict(
                    values=list(tabla.columns),
                    font_family=_st.CUSTOM_FONT,
                    font_size=13,
                    font_color="#ECEFF4",
                    fill_color="#5E81AC",
                    align="center",
                    height=26,
                ),
                cells=dict(
                    values=[tabla.iloc[:, 0], tabla.iloc[:, 1], tabla.iloc[:, 2]],
                    font_family=_st.CUSTOM_FONT,
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

if all(st.session_state.files):
    _st.googlef_text("📊 Laboratory Graphs", key="h3", color="#434C5E")

    graflabo_escogido = st.selectbox(
        "Seleccione uno de los gráficos obtenidos en laboratorio.",
        (
            "Deformación Axial [d] - Fuerza Axial [ΔP]",
            "Deformación Axial [d] - Volumen de Buretas [Vb]",
        ),
    )

    colores = ["#8FBCBB", "#88C0D0", "#81A1C1", "#5E81AC"]

    fig = go.Figure()

    if graflabo_escogido == "Deformación Axial [d] - Fuerza Axial [ΔP]":
        ptos_max = []

        for ensayo in st.session_state.tests:
            ptos_max.append(
                ensayo.loc[ensayo.iloc[:, 1] == max(ensayo.iloc[:, 1])].iloc[0]
            )

        for j, ensayo in enumerate(st.session_state.tests):
            fig.add_trace(
                go.Scatter(
                    x=ensayo["d (mm)"],
                    y=ensayo["ΔP"],
                    name="Test #{}".format(j + 1),
                    line=dict(color=colores[j + 1]),
                    text="σ₀ = {}".format(st.session_state.esf_conf[j]),
                ),
            )

        for j, pair in enumerate(ptos_max):
            fig.add_trace(
                go.Scatter(
                    x=np.array(pair[0]),
                    y=np.array(pair[1]),
                    name="Test #{}".format(j + 1),
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
            title_text="Fuerza Axial, ΔP",
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
        for j, ensayo in enumerate(st.session_state.tests):
            fig.add_trace(
                go.Scatter(
                    x=ensayo["d (mm)"],
                    y=ensayo["Vb (cm³)"],
                    name="Test #{}".format(j + 1),
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
            title_text="Volumen de Buretas, Vb (cm³)",
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
        font=dict(family=_st.CUSTOM_FONT, color="#5E81AC"),
        showlegend=False,
        template="seaborn",
        paper_bgcolor="#ECEFF4",
        plot_bgcolor="#E5E9F0",
        margin=dict(l=0, r=0, b=0, t=0),
        hoverlabel=dict(
            font_family=_st.CUSTOM_FONT,
        ),
    )

    st.plotly_chart(fig, config={"displayModeBar": False})
else:
    "⌛ Esperando registros de laboratorio..."


############################################
# Cuarto bloque: Preprocesamiento de Datos #
############################################

_st.googlef_text("4️⃣ Data Preprocesing", key="h2")

if all(st.session_state.files):
    hf = [h - hc / 10 for hc in h_cons]
    hf_dict = dict(zip(["Test #1", "Test #2", "Test #3"], hf))

    vf = [V - vc for vc in v_cons]
    vf_dict = dict(zip(["Test #1", "Test #2", "Test #3"], vf))

    for i, ensayo in enumerate(st.session_state.tests):
        ensayo["ε (%)"] = ensayo["d (mm)"] / hf[i] * 10
        ensayo["ΔV (%)"] = (ensayo["Vb (cm³)"][0] - ensayo["Vb (cm³)"]) / vf[i] * 100
        ensayo["Ac (cm²)"] = (vf[i] - ensayo["Vb (cm³)"]) / (
            hf[i] - ensayo["d (mm)"] / 10
        )
        ensayo["Δσ ({0})".format(unit)] = ensayo["ΔP"] / ensayo["Ac (cm²)"] * 10**4

    _st.googlef_text("🏴 Geometry | Consolidation", key="h3", color="#434C5E")

    tabla_preprocs = st.selectbox(
        "Seleccione uno de los registros de laboratorio.",
        ("Test #1", "Test #2", "Test #3"),
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

    _st.googlef_text("📋 Preprocesing Results", key="h3", color="#434C5E")

    tables = {
        "Test #1": st.session_state.tests[0].iloc[:, [3, 4, 5, 6]],
        "Test #2": st.session_state.tests[1].iloc[:, [3, 4, 5, 6]],
        "Test #3": st.session_state.tests[2].iloc[:, [3, 4, 5, 6]],
    }
    tabla = tables[tabla_preprocs]

    fig = go.Figure(
        data=[
            go.Table(
                header=dict(
                    values=list(tabla.columns),
                    font_family=_st.CUSTOM_FONT,
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
                    font_family=_st.CUSTOM_FONT,
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

    _st.googlef_text("📊 Preprocesing Results", key="h3", color="#434C5E")

    grafpreprosc_escogido = st.selectbox(
        "Seleccione uno de los gráficos obtenidos del preprocesamiento.",
        (
            "Def. Axial Unitaria [ε] - Deviatoric Stress [Δσ]",
            "Def. Axial Unitaria [ε] - Def. Volumétrica Unitaria [ΔV]",
        ),
    )

    ptos_max = []

    for ensayo in st.session_state.tests:
        ptos_max.append(ensayo.loc[ensayo.iloc[:, 6] == max(ensayo.iloc[:, 6])].iloc[0])

    fig2 = go.Figure()

    if grafpreprosc_escogido == "Def. Axial Unitaria [ε] - Deviatoric Stress [Δσ]":

        for j, ensayo in enumerate(st.session_state.tests):
            fig2.add_trace(
                go.Scatter(
                    x=ensayo["ε (%)"],
                    y=ensayo["Δσ ({0})".format(unit)],
                    name="Test #{}".format(j + 1),
                    line=dict(color=colores[j + 1]),
                    text="σ₀ = {}".format(st.session_state.esf_conf[j]),
                ),
            )

        for j, pair in enumerate(ptos_max):
            fig2.add_trace(
                go.Scatter(
                    x=np.array(pair[3]),
                    y=np.array(pair[6]),
                    name="Test #{}".format(j + 1),
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
            title_text="Axial Strain, ε (%)",
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
            title_text="Deviatoric Stress, Δσ ({0})".format(unit),
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
        for j, ensayo in enumerate(st.session_state.tests):
            fig2.add_trace(
                go.Scatter(
                    x=ensayo["ε (%)"],
                    y=ensayo["ΔV (%)"],
                    name="Test #{}".format(j + 1),
                    line=dict(color=colores[j + 1]),
                    text="σ₀ = {}".format(st.session_state.esf_conf[j]),
                ),
            )

        fig2.update_xaxes(
            title_text="Axial Strain, ε (%)",
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
            title_text="Volumetric Strain, ΔV (%)",
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
        font=dict(family=_st.CUSTOM_FONT, color="#5E81AC"),
        showlegend=False,
        template="seaborn",
        paper_bgcolor="#ECEFF4",
        plot_bgcolor="#E5E9F0",
        margin=dict(l=0, r=0, b=0, t=0),
        hoverlabel=dict(
            font_family=_st.CUSTOM_FONT,
        ),
    )

    st.plotly_chart(fig2, config={"displayModeBar": False})


else:
    "⌛ Esperando registros de laboratorio..."

#############################################
# Quinto bloque: Procesamiento y Resultados #
#############################################

_st.googlef_text("5️⃣ Procesing and Results", key="h2")

if all(st.session_state.files):

    # Trayectorias de Esfuerzos #
    #############################

    _st.googlef_text("📈 Stress Paths", key="h3", color="#434C5E")

    convencion = st.radio(
        "Escoga una convención para el cálculo de las invariantes de esfuerzos.",
        ("Massachusetts Institute of Technology", "University of Cambridge"),
    )

    if convencion == "University of Cambridge":
        for i, ensayo in enumerate(st.session_state.tests):
            ensayo["p' ({0})".format(unit)] = (
                st.session_state.esf_conf[i] + ensayo["Δσ ({0})".format(unit)] / 3
            )
            ensayo["q ({0})".format(unit)] = ensayo["Δσ ({0})".format(unit)]

            ensayo["p' [M.I.T]"] = (
                st.session_state.esf_conf[i] + ensayo["Δσ ({0})".format(unit)] / 2
            )
            ensayo["q [M.I.T]"] = ensayo["Δσ ({0})".format(unit)] / 2
    else:
        for i, ensayo in enumerate(st.session_state.tests):
            ensayo["p' ({0})".format(unit)] = (
                st.session_state.esf_conf[i] + ensayo["Δσ ({0})".format(unit)] / 2
            )
            ensayo["q ({0})".format(unit)] = ensayo["Δσ ({0})".format(unit)] / 2

    fig3 = go.Figure()

    for i, ensayo in enumerate(st.session_state.tests):
        fig3.add_trace(
            go.Scatter(
                x=ensayo["p' ({0})".format(unit)],
                y=ensayo["q ({0})".format(unit)],
                name="Test #{}".format(i + 1),
                line=dict(color=colores[i + 1]),
                text="σ₀ = {}".format(st.session_state.esf_conf[i]),
            )
        )
        fig3.add_trace(
            go.Scatter(
                x=np.array(ensayo["p' ({0})".format(unit)].iloc[-1]),
                y=np.array(ensayo["q ({0})".format(unit)].iloc[-1]),
                name="Test #{}".format(i + 1),
                marker_size=7,
                marker_color=colores[i + 1],
                text="σ₀ = {}".format(st.session_state.esf_conf[i]),
            )
        )

    if convencion == "University of Cambridge":
        p_falla_MIT = np.array(
            [ensayo["p' [M.I.T]"].iloc[-1] for ensayo in st.session_state.tests]
        ).reshape(-1, 1)
        q_falla_MIT = np.array(
            [ensayo["q [M.I.T]"].iloc[-1] for ensayo in st.session_state.tests]
        )

        modelo_MIT = LinearRegression()
        modelo_MIT.fit(p_falla_MIT, q_falla_MIT)

    p_falla = np.array(
        [ensayo["p' ({0})".format(unit)].iloc[-1] for ensayo in st.session_state.tests]
    ).reshape(-1, 1)
    q_falla = np.array(
        [ensayo["q ({0})".format(unit)].iloc[-1] for ensayo in st.session_state.tests]
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
            name="Mod. Envelope",
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
        title_text="p' ({0})".format(unit),
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
        title_text="q ({0})".format(unit),
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
        font=dict(family=_st.CUSTOM_FONT, color="#5E81AC"),
        showlegend=False,
        template="seaborn",
        paper_bgcolor="#ECEFF4",
        plot_bgcolor="#E5E9F0",
        margin=dict(l=0, r=0, b=0, t=0),
        hoverlabel=dict(font_family=_st.CUSTOM_FONT),
    )

    st.write("")
    st.plotly_chart(fig3, config={"displayModeBar": False})

    # Círculos de Mohr #
    ####################

    _st.googlef_text("📉 Mohr Circles", key="h3", color="#434C5E")

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
        title_text="σ' ({0})".format(unit),
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
        title_text="τ ({0})".format(unit),
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
        font=dict(family=_st.CUSTOM_FONT, color="#5E81AC"),
        showlegend=False,
        template="seaborn",
        paper_bgcolor="#ECEFF4",
        plot_bgcolor="#E5E9F0",
        margin=dict(l=0, r=0, b=0, t=0),
        hoverlabel=dict(font_family=_st.CUSTOM_FONT),
    )

    st.write("")
    st.plotly_chart(fig4, config={"displayModeBar": False})

    # Resumen de Resultados #
    ########################

    _st.googlef_text("🎈 Summary of Results", key="h3", color="#434C5E")

    resultados = pd.DataFrame(
        {
            "Tests": ["Test #1", "Test #2", "Test #3"],
            "σr' ({0})".format(unit): [
                st.session_state["esf_conf"][0],
                st.session_state["esf_conf"][1],
                st.session_state["esf_conf"][2],
            ],
            "Δσf ({0})".format(unit): [
                max(st.session_state.tests[0]["Δσ ({0})".format(unit)]),
                max(st.session_state.tests[1]["Δσ ({0})".format(unit)]),
                max(st.session_state.tests[2]["Δσ ({0})".format(unit)]),
            ],
        }
    )

    fig5 = go.Figure(
        data=[
            go.Table(
                header=dict(
                    values=list(resultados.columns),
                    font_family=_st.CUSTOM_FONT,
                    font_size=13,
                    font_color="#ECEFF4",
                    fill_color=["#4C566A", "#81A1C1", "#81A1C1"],
                    align="center",
                    height=30,
                ),
                cells=dict(
                    values=[
                        resultados.iloc[:, 0],
                        resultados.iloc[:, 1],
                        resultados.iloc[:, 2],
                    ],
                    font_family=_st.CUSTOM_FONT,
                    font_size=13,
                    fill_color=["#D8DEE9", "#E5E9F0", "#E5E9F0"],
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
    st.write("⌛ Esperando registros de laboratorio...")
