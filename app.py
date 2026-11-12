import os
from datetime import datetime

import streamlit as st
import pandas as pd
import plotly.express as px
import scipy.stats as stats
import numpy as np
from sqlalchemy import create_engine, text

# ----------------------------------------------------------------------
# CONFIGURACIÓN SEGURA - Base de Datos MySQL
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
# CONEXIÓN Y UTILIDADES
# ----------------------------------------------------------------------
def obtener_conexion():
    cadena_conexion = f"mysql+mysqlconnector://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
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
        st.error(f"Error de conexión o lectura en la base de datos MySQL: {e}")
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

def intervalo_wilson(k, n, z=1.96):
    p = k / n
    den = 1 + z**2 / n
    centro = (p + z**2 / (2 * n)) / den
    mitad = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / den
    return centro - mitad, centro + mitad

def es_optimo(serie):
    return serie.str.contains("OPTIMO", case=False, na=False)

def renderizar_tabla_html_segura(df_input):
    """Renderiza tablas en HTML puro, previniendo fallos en el motor de React."""
    html_table = df_input.to_html(classes="tabla-estilizada", escape=False)
    style = """
    <style>
        .tabla-estilizada {
            width: 100%;
            border-collapse: collapse;
            margin: 10px 0;
            font-size: 14px;
            color: #ffffff;
            background-color: #1e1e1e;
            border-radius: 6px;
            overflow: hidden;
        }
        .tabla-estilizada th {
            background-color: #333333;
            color: #2ecc71;
            text-align: left;
            padding: 8px 12px;
            border-bottom: 2px solid #444;
        }
        .tabla-estilizada td {
            padding: 8px 12px;
            border-bottom: 1px solid #2a2a2a;
        }
    </style>
    """
    st.markdown(style + html_table, unsafe_allow_html=True)

# ----------------------------------------------------------------------
# COMPONENTE VISUAL: SEMÁFORO
# ----------------------------------------------------------------------
def mostrar_semaforo_visual(tiempo_promedio, tiempo_ideal=5.0, tolerancia=1.2):
    umbral_amarillo = tiempo_ideal * tolerancia

    if tiempo_promedio <= tiempo_ideal:
        estado = "verde"
        titulo = "🟢 ESTADO ÓPTIMO"
        desc = f"El tiempo promedio (**{tiempo_promedio:.2f} min**) cumple la meta (≤ {tiempo_ideal:.1f} min)."
    elif tiempo_promedio <= umbral_amarillo:
        estado = "amarillo"
        titulo = "🟡 ESTADO DE PRECAUCIÓN"
        desc = f"El tiempo promedio (**{tiempo_promedio:.2f} min**) excede ligeramente la meta ideal."
    else:
        estado = "rojo"
        titulo = "🔴 ESTADO CRÍTICO"
        desc = f"El tiempo promedio (**{tiempo_promedio:.2f} min**) supera el límite aceptable."

    c_rojo = "#ff4d4d" if estado == "rojo" else "#4d1a1a"
    c_amarillo = "#ffcc00" if estado == "amarillo" else "#4d4d00"
    c_verde = "#2ecc71" if estado == "verde" else "#143d21"

    html_semaforo = f"""
    <div style="background-color: #1e1e1e; padding: 15px; border-radius: 10px; border: 1px solid #333; margin-bottom: 15px; display: flex; align-items: center;">
        <div style="display: flex; gap: 10px; background: #111; padding: 8px 14px; border-radius: 20px;">
            <div style="width: 20px; height: 20px; background-color: {c_rojo}; border-radius: 50%;"></div>
            <div style="width: 20px; height: 20px; background-color: {c_amarillo}; border-radius: 50%;"></div>
            <div style="width: 20px; height: 20px; background-color: {c_verde}; border-radius: 50%;"></div>
        </div>
        <div style="margin-left: 15px;">
            <h4 style="margin: 0; color: #fff;">{titulo}</h4>
            <p style="margin: 0; color: #ccc; font-size: 13px;">{desc}</p>
        </div>
    </div>
    """
    st.markdown(html_semaforo, unsafe_allow_html=True)

# ----------------------------------------------------------------------
# VISTAS
# ----------------------------------------------------------------------
def vista_registro():
    st.title("Registrar Atención")

    with st.form("registro_form", clear_on_submit=True):
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
            st.success("¡Registro guardado en MySQL con éxito!")
            st.rerun()
        except Exception as e:
            st.error(f"Error al guardar: {e}")

    st.divider()
    st.subheader("Últimos registros")
    mostrar_ultimos()

def mostrar_ultimos():
    df = cargar_datos()
    st.caption(f"Total registrados: {len(df)}")

    if not df.empty:
        df_visual = (
            df[["GRUPO DE SERVICIO", "SERVICIO", "Minutos", "SEXO", "SATISFACCION"]]
            .tail(5)
            .iloc[::-1]
            .copy()
        )
        df_visual["TIEMPO (MM:SS)"] = df_visual["Minutos"].apply(_formatear_minutos_a_mmss)
        df_visual = df_visual[["GRUPO DE SERVICIO", "SERVICIO", "TIEMPO (MM:SS)", "SEXO", "SATISFACCION"]]
        renderizar_tabla_html_segura(df_visual)

def vista_dashboard():
    st.title("Resumen del Negocio")
    df = cargar_datos()

    if df.empty:
        st.info("Sin registros en la base de datos.")
        return

    tiempo_promedio = df["Minutos"].mean()
    felices = int(es_optimo(df["SATISFACCION"]).sum())
    porcentaje_felices = (felices / len(df)) * 100

    mostrar_semaforo_visual(tiempo_promedio=tiempo_promedio, tiempo_ideal=REF_TIEMPO)

    # Tarjetas Métricas en HTML
    metrics_html = f"""
    <div style="display: flex; gap: 10px; margin-bottom: 20px;">
        <div style="flex: 1; background: #1e1e1e; padding: 12px; border-radius: 8px; text-align: center;">
            <span style="color: #888; font-size: 12px;">Clientes Atendidos</span>
            <h2 style="margin: 5px 0 0 0; color: #fff;">{len(df)}</h2>
        </div>
        <div style="flex: 1; background: #1e1e1e; padding: 12px; border-radius: 8px; text-align: center;">
            <span style="color: #888; font-size: 12px;">Tiempo Promedio</span>
            <h2 style="margin: 5px 0 0 0; color: #fff;">{_formatear_minutos_a_mmss(tiempo_promedio)} m</h2>
        </div>
        <div style="flex: 1; background: #1e1e1e; padding: 12px; border-radius: 8px; text-align: center;">
            <span style="color: #888; font-size: 12px;">Satisfacción</span>
            <h2 style="margin: 5px 0 0 0; color: #fff;">{porcentaje_felices:.0f}%</h2>
        </div>
    </div>
    """
    st.markdown(metrics_html, unsafe_allow_html=True)
    st.divider()

    st.subheader("Tiempos por Grupo")
    df_grupos = df.groupby("GRUPO DE SERVICIO")["Minutos"].mean().reset_index()
    df_grupos["GRUPO DE SERVICIO"] = "Grupo " + df_grupos["GRUPO DE SERVICIO"].astype(int).astype(str)
    df_grupos["Tiempo_Texto"] = df_grupos["Minutos"].apply(_formatear_minutos_a_mmss)

    fig1 = px.bar(df_grupos, x="GRUPO DE SERVICIO", y="Minutos", color="GRUPO DE SERVICIO", text="Tiempo_Texto")
    st.plotly_chart(fig1, use_container_width=True)

def vista_inferencial():
    st.title("Análisis Estadístico")
    df = cargar_datos()
    n_total = len(df)

    if n_total < 2:
        st.warning("Se requieren al menos 2 registros.")
        return

    st.subheader("1. Estimación General")
    mean_time = df["Minutos"].mean()
    std_time = df["Minutos"].std()
    
    ci_time = stats.t.interval(0.95, n_total - 1, loc=mean_time, scale=std_time / np.sqrt(n_total)) if std_time > 0 else (mean_time, mean_time)
    felices = int(es_optimo(df["SATISFACCION"]).sum())

    lo = stats.beta.ppf(0.025, felices, n_total - felices + 1) if felices > 0 else 0.0
    hi = stats.beta.ppf(0.975, felices + 1, n_total - felices) if felices < n_total else 1.0

    st.info(f"🕒 **Tiempo estimado (95% confianza):** {_formatear_minutos_a_mmss(ci_time[0])} a {_formatear_minutos_a_mmss(ci_time[1])} min.")
    st.info(f"😊 **Satisfacción estimada (95% confianza):** {lo*100:.1f}% a {hi*100:.1f}%.")

def vista_excel():
    st.title("Modo Ingeniero (Registros BD)")
    df = cargar_datos()
    st.caption(f"Total registros: {len(df)}")

    renderizar_tabla_html_segura(df)

    if not df.empty:
        csv_data = df.to_csv(sep=";", index=True, encoding="utf-8-sig").encode("utf-8-sig")
        st.download_button(
            "⬇️ Descargar Reporte (CSV Excel)",
            data=csv_data,
            file_name="Datos_Imprenta_MySQL.csv",
            mime="text/csv",
        )

# Enrutador
if menu == "⏱️ Registrar un cliente":
    vista_registro()
elif menu == "📊 Dashboard Descriptivo":
    vista_dashboard()
elif menu == "📈 Análisis Inferencial":
    vista_inferencial()
else:
    vista_excel()