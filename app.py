import os
from datetime import datetime

import streamlit as st
import pandas as pd
import plotly.express as px
import scipy.stats as stats
import numpy as np
from sqlalchemy import create_engine, text

# ----------------------------------------------------------------------
# CONFIGURACIÓN Y CONEXIÓN
# ----------------------------------------------------------------------
def cargar_configuracion_db():
    try:
        if "mysql" in st.secrets:
            return (
                st.secrets["mysql"]["host"],
                st.secrets["mysql"]["user"],
                st.secrets["mysql"]["password"],
                st.secrets["mysql"]["database"],
                int(st.secrets["mysql"].get("port", 3306)),
            )
    except Exception:
        pass

    return (
        os.getenv("DB_HOST", "localhost"),
        os.getenv("DB_USER", "root"),
        os.getenv("DB_PASS", ""),
        os.getenv("DB_NAME", "mi_bd"),
        int(os.getenv("DB_PORT", 3306)),
    )

DB_HOST, DB_USER, DB_PASS, DB_NAME, DB_PORT = cargar_configuracion_db()
DB_TABLE = "imprenta_2"

st.set_page_config(page_title="Imprenta Flores", layout="centered")

# Menú lateral
st.sidebar.title("Imprenta Flores")
menu = st.sidebar.radio(
    "¿Qué deseas hacer?",
    [
        "⏱️ Registrar un cliente",
        "📊 Dashboard Descriptivo",
        "📈 Análisis Inferencial",
        "🛠️ Modo Ingeniero (Ver BD)",
    ],
    key="menu_navegacion"
)

st.sidebar.markdown("---")
st.sidebar.subheader("⚙️ Configuración de Metas")

REF_TIEMPO = st.sidebar.number_input(
    "Meta máx. de tiempo (min)",
    min_value=0.5,
    max_value=60.0,
    value=5.0,
    step=0.5,
    key="ref_tiempo",
)

REF_SAT = st.sidebar.number_input(
    "Meta mín. satisfacción (%)",
    min_value=1,
    max_value=100,
    value=90,
    step=5,
    key="ref_sat",
)

# ----------------------------------------------------------------------
# BASE DE DATOS
# ----------------------------------------------------------------------
def obtener_conexion():
    cadena_conexion = f"mysql+pymysql://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
    return create_engine(cadena_conexion)

def _convertir_a_minutos(t):
    try:
        if isinstance(t, pd.Timedelta):
            return t.total_seconds() / 60.0
        return t.hour * 60 + t.minute + t.second / 60.0
    except Exception:
        partes = str(t).split(":")
        return int(partes[0]) * 60 + int(partes[1]) + float(partes[2]) / 60.0

def _formatear_minutos_a_mmss(minutos_decimal):
    try:
        total_segundos = int(round(float(minutos_decimal) * 60))
        m, s = divmod(total_segundos, 60)
        return f"{m:02d}:{s:02d}"
    except Exception:
        return str(minutos_decimal)

def cargar_datos():
    try:
        engine = obtener_conexion()
        query = f"SELECT codigo, grupo, servicio, tiempo, sexo, satisfaccion, fecha_actual FROM {DB_TABLE} ORDER BY codigo ASC"
        df = pd.read_sql(query, con=engine)

        if df.empty:
            return pd.DataFrame(
                columns=[
                    "CODIGO", "GRUPO DE SERVICIO", "SERVICIO", "TIEMPO_EXCEL",
                    "SEXO", "SATISFACCION", "Minutos", "FECHA_ACTUAL"
                ]
            )

        df.rename(
            columns={
                "codigo": "CODIGO",
                "grupo": "GRUPO DE SERVICIO",
                "servicio": "SERVICIO",
                "tiempo": "TIEMPO_EXCEL",
                "sexo": "SEXO",
                "satisfaccion": "SATISFACCION",
                "fecha_actual": "FECHA_ACTUAL",
            },
            inplace=True,
        )

        df["Minutos"] = df["TIEMPO_EXCEL"].apply(_convertir_a_minutos)
        df["SATISFACCION"] = df["SATISFACCION"].astype(str).str.strip()
        df.set_index("CODIGO", inplace=True)
        return df

    except Exception as e:
        st.error(f"Error al conectar con MySQL: {e}")
        st.stop()

def guardar_registro(grupo, servicio, sexo, minutos, satisfaccion):
    total_seg = int(round(minutos * 60))
    m, s = divmod(total_seg, 60)
    h, m = divmod(m, 60)
    tiempo_formateado = f"{h:02d}:{m:02d}:{s:02d}"
    fecha_hoy = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    engine = obtener_conexion()
    query = text(f"""
        INSERT INTO {DB_TABLE} (grupo, servicio, tiempo, sexo, satisfaccion, fecha_actual)
        VALUES (:grupo, :servicio, :tiempo, :sexo, :satisfaccion, :fecha_actual)
    """)

    with engine.begin() as conn:
        conn.execute(
            query,
            {
                "grupo": grupo,
                "servicio": servicio,
                "tiempo": tiempo_formateado,
                "sexo": sexo,
                "satisfaccion": satisfaccion,
                "fecha_actual": fecha_hoy,
            },
        )

def es_optimo(serie):
    return serie.str.contains("OPTIMO", case=False, na=False)

# ----------------------------------------------------------------------
# CONTENEDOR PRINCIPAL ISOLADO
# ----------------------------------------------------------------------
area_trabajo = st.container()

with area_trabajo:
    if menu == "⏱️ Registrar un cliente":
        st.title("Registrar Atención")

        with st.form("form_registro", clear_on_submit=True):
            grupo = st.radio("Tipo de servicio:", ["Grupo 1 (Copias/Impresiones)", "Grupo 2 (Trámites/Pasaporte)"])
            servicio = st.text_input("Servicio exacto:")
            sexo = st.radio("Sexo:", ["F", "M"], horizontal=True)
            tiempo = st.number_input("Minutos transcurridos:", min_value=0.1, value=1.0, step=0.5)
            satisfaccion = st.radio("Satisfacción:", ["Sí, OPTIMO 😊", "NO, hubo molestias 😞"])
            enviar = st.form_submit_button("Guardar Registro ✔️", use_container_width=True)

        if enviar:
            try:
                guardar_registro(
                    grupo=1 if "Grupo 1" in grupo else 2,
                    servicio=servicio.strip() if servicio.strip() else "Varios",
                    sexo=sexo,
                    minutos=tiempo,
                    satisfaccion="OPTIMO" if "Sí" in satisfaccion else "REGULAR",
                )
                st.success("¡Registro guardado con éxito!")
                st.rerun()
            except Exception as e:
                st.error(f"Error al guardar: {e}")

        st.divider()
        st.subheader("Últimos registros")
        df = cargar_datos()
        if not df.empty:
            df_vis = df[["GRUPO DE SERVICIO", "SERVICIO", "Minutos", "SEXO", "SATISFACCION"]].tail(5).iloc[::-1]
            st.dataframe(df_vis, use_container_width=True)

    elif menu == "📊 Dashboard Descriptivo":
        st.title("Resumen del Negocio")
        df = cargar_datos()

        if df.empty:
            st.info("Sin registros en la base de datos.")
        else:
            tiempo_prom = df["Minutos"].mean()
            felices = int(es_optimo(df["SATISFACCION"]).sum())
            pct_felices = (felices / len(df)) * 100

            col1, col2, col3 = st.columns(3)
            col1.metric("Clientes Atendidos", len(df))
            col2.metric("Tiempo Promedio", f"{_formatear_minutos_a_mmss(tiempo_prom)} m")
            col3.metric("Satisfacción", f"{pct_felices:.0f}%")

            st.divider()
            st.subheader("Tiempos por Grupo")
            df_grupos = df.groupby("GRUPO DE SERVICIO")["Minutos"].mean().reset_index()
            df_grupos["GRUPO DE SERVICIO"] = "Grupo " + df_grupos["GRUPO DE SERVICIO"].astype(int).astype(str)

            fig = px.bar(df_grupos, x="GRUPO DE SERVICIO", y="Minutos", color="GRUPO DE SERVICIO")
            st.plotly_chart(fig, use_container_width=True)

    elif menu == "📈 Análisis Inferencial":
        st.title("Análisis Estadístico")
        df = cargar_datos()
        if len(df) < 2:
            st.warning("Se requieren al menos 2 registros.")
        else:
            mean_t = df["Minutos"].mean()
            std_t = df["Minutos"].std()
            ci = stats.t.interval(0.95, len(df)-1, loc=mean_t, scale=std_t/np.sqrt(len(df))) if std_t > 0 else (mean_t, mean_t)

            st.info(f"🕒 **Tiempo estimado (95% confianza):** {_formatear_minutos_a_mmss(ci[0])} a {_formatear_minutos_a_mmss(ci[1])} min.")

    else:
        st.title("Modo Ingeniero (BD MySQL)")
        df = cargar_datos()
        st.dataframe(df, use_container_width=True)

        if not df.empty:
            csv_data = df.to_csv(sep=";", index=True, encoding="utf-8-sig").encode("utf-8-sig")
            st.download_button(
                "⬇️ Descargar Reporte (CSV para Excel)",
                data=csv_data,
                file_name="Datos_Imprenta_MySQL.csv",
                mime="text/csv",
                key="btn_csv_export"
            )