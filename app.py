import streamlit as st
import pandas as pd
import joblib

from config.curso_area import CURSOS

# Cargar modelo
model = joblib.load("models/model.pkl")

# Inicializar historial
if "historial" not in st.session_state:
    st.session_state.historial = []

st.title("🎓 Predicción Académica por Áreas")

# Datos generales
st.subheader("Factores Académicos")

studytime = st.number_input(
    "Horas de estudio semanales",
    min_value=0,
    value=10
)

failures = st.number_input(
    "Cantidad de cursos reprobados",
    min_value=0,
    value=0
)

absences = st.number_input(
    "Cantidad de ausencias",
    min_value=0,
    value=0
)

# Historial académico
st.subheader("📚 Historial Académico")

curso = st.selectbox(
    "Seleccione un curso",
    list(CURSOS.keys())
)

nota = st.number_input(
    "Nota final obtenida",
    min_value=0,
    max_value=100,
    value=70
)

if st.button("Agregar curso"):

    st.session_state.historial.append({
        "curso": curso,
        "area": CURSOS[curso]["area"],
        "creditos": CURSOS[curso]["creditos"],
        "nota": nota
    })

    st.success("Curso agregado correctamente")

# Mostrar historial
if st.session_state.historial:

    df_historial = pd.DataFrame(
        st.session_state.historial
    )

    st.subheader("📋 Cursos Ingresados")
    st.dataframe(
        df_historial,
        use_container_width=True
    )

    # Calcular promedio ponderado por créditos
    df_historial["nota_x_creditos"] = (
        df_historial["nota"] *
        df_historial["creditos"]
    )

    resumen_areas = (
        df_historial
        .groupby("area")
        .agg({
            "nota_x_creditos": "sum",
            "creditos": "sum"
        })
        .reset_index()
    )

    resumen_areas["promedio_area"] = (
        resumen_areas["nota_x_creditos"] /
        resumen_areas["creditos"]
    )

    st.subheader("📊 Rendimiento por Área")
    st.dataframe(
        resumen_areas[
            [
                "area",
                "promedio_area",
                "creditos"
            ]
        ],
        use_container_width=True
    )

#Predicción
if st.button("Predecir por áreas"):

    if not st.session_state.historial:

        st.warning(
            "Debe ingresar al menos un curso."
        )

    else:

        # Predicción general usando IA
        pred_general = model.predict([[
            studytime,
            failures,
            absences
        ]])[0]

        st.subheader("🤖 Predicción General")

        st.success(
            f"Rendimiento estimado: {pred_general}"
        )

        df_historial = pd.DataFrame(
            st.session_state.historial
        )

        df_historial["nota_x_creditos"] = (
            df_historial["nota"] *
            df_historial["creditos"]
        )

        resumen_areas = (
            df_historial
            .groupby("area")
            .agg({
                "nota_x_creditos": "sum",
                "creditos": "sum"
            })
            .reset_index()
        )

        resumen_areas["promedio_area"] = (
            resumen_areas["nota_x_creditos"] /
            resumen_areas["creditos"]
        )

        st.subheader("🎯 Predicción por Área")

        for _, fila in resumen_areas.iterrows():

            promedio = fila["promedio_area"]

            # Ajuste usando historial del área
            if promedio >= 85:
                categoria = "Alto 🟢"

            elif promedio >= 70:

                if pred_general == "Alto":
                    categoria = "Alto 🟢"
                else:
                    categoria = "Medio 🟡"

            else:

                if pred_general == "Bajo":
                    categoria = "Bajo 🔴"
                else:
                    categoria = "Medio 🟡"

            st.write(
                f"**{fila['area']}** → {categoria}"
            )
            
            if categoria == "Alto 🟢":
                st.success(
                    "Recomendación: Mantenga sus hábitos actuales de estudio."
                )

            elif categoria == "Medio 🟡":
                st.warning(
                    "Recomendación: Dedique más tiempo al estudio y refuerce los temas con mayor dificultad."
                )

            else:
                st.error(
                    "Recomendación: Considere tutorías, grupos de estudio o apoyo académico para fortalecer esta área."
                )