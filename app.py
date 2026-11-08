import os
import threading
from datetime import datetime, time as dtime

import streamlit as st
import pandas as pd
import plotly.express as px
import scipy.stats as stats
import numpy as np
from openpyxl import load_workbook

# ----------------------------------------------------------------------
# CONFIGURACIÓN - Base de datos
# ----------------------------------------------------------------------
ARCHIVO = "Datos.xlsx"   # Base de datos (debe estar junto a este archivo)
REFRESCO = 5             # Cada cuántos segundos se actualizan las pantallas

st.set_page_config(page_title="Imprenta Flores", layout="centered")

st.sidebar.title("Imprenta Floresxy")
menu = st.sidebar.radio("¿Qué deseas hacer?", [
    "⏱️ Registrar un cliente",
    "📊 Dashboard Descriptivo",
    "📈 Análisis Inferencial",
    "🛠️ Modo Ingeniero (Ver Excel)"
])

# ----------------------------------------------------------------------
# CONFIGURACIÓN GLOBAL DE METAS (CON LÍMITES LÓGICOS)
# ----------------------------------------------------------------------
st.sidebar.markdown("---")
st.sidebar.subheader("⚙️ Configuración de Metas")

# Restringido: El tiempo mínimo útil es 0.5 min (sin negativos ni 0)
REF_TIEMPO = st.sidebar.number_input(
    "Meta máx. de tiempo (min)", 
    min_value=0.5, 
    max_value=60.0, 
    value=5.0, 
    step=0.5, 
    key="ref_tiempo"
)

# Restringido: El porcentaje de satisfacción va de 1% a 100%
REF_SAT = st.sidebar.number_input(
    "Meta mín. satisfacción (%)", 
    min_value=1, 
    max_value=100, 
    value=90, 
    step=5, 
    key="ref_sat"
)


# ----------------------------------------------------------------------
# BASE DE DATOS (EXCEL)
# ----------------------------------------------------------------------
@st.cache_resource
def obtener_candado():
    """Un solo candado compartido por todas las sesiones, para que dos
    registros simultáneos no se pisen al escribir el Excel."""
    return threading.Lock()


def _convertir_a_minutos(t):
    try:
        return t.hour * 60 + t.minute + t.second / 60
    except Exception:
        partes = str(t).split(':')
        return int(partes[0]) * 60 + int(partes[1]) + float(partes[2]) / 60


def _formatear_minutos_a_mmss(minutos_decimal):
    """Convierte un valor decimal de minutos a formato 'MM:SS' para mostrarse amigable."""
    try:
        total_segundos = int(round(float(minutos_decimal) * 60))
        m, s = divmod(total_segundos, 60)
        return f"{m:02d}:{s:02d}"
    except Exception:
        return str(minutos_decimal)


@st.cache_data
def _leer_excel(ruta, fecha_modificacion):
    """La caché depende de la fecha de modificación del archivo: si el Excel
    cambia (por la app o editado a mano), se vuelve a leer automáticamente."""
    df = pd.read_excel(ruta)
    df = df.iloc[:, :5]
    df.columns = ['GRUPO DE SERVICIO', 'SERVICIO', 'TIEMPO_EXCEL', 'SEXO', 'SATISFACCION']
    df = df.dropna(subset=['GRUPO DE SERVICIO'])
    df['Minutos'] = df['TIEMPO_EXCEL'].apply(_convertir_a_minutos)
    df['SATISFACCION'] = df['SATISFACCION'].astype(str).str.strip()
    df.index = range(1, len(df) + 1)
    return df


def cargar_datos():
    try:
        return _leer_excel(ARCHIVO, os.path.getmtime(ARCHIVO))
    except FileNotFoundError:
        st.error(f"No encuentro el archivo {ARCHIVO}. Debe estar en la misma carpeta que app.py.")
        st.stop()
    except Exception as e:
        st.warning(f"No pude leer el Excel en este momento (¿se está guardando?). Reintentando... ({e})")
        st.stop()


def guardar_registro(grupo, servicio, sexo, minutos, satisfaccion):
    """Agrega una fila al final del Excel sin reescribir todo el archivo."""
    total_seg = int(round(minutos * 60))
    m, s = divmod(total_seg, 60)
    h, m = divmod(m, 60)

    with obtener_candado():
        wb = load_workbook(ARCHIVO)
        ws = wb.active
        fila = ws.max_row
        while fila > 1 and ws.cell(fila, 1).value is None:
            fila -= 1
        fila += 1

        ws.cell(fila, 1, grupo)
        ws.cell(fila, 2, servicio)
        celda_tiempo = ws.cell(fila, 3, dtime(h, m, s))
        celda_tiempo.number_format = "h:mm:ss"
        ws.cell(fila, 4, sexo)
        ws.cell(fila, 5, satisfaccion)
        wb.save(ARCHIVO)


def intervalo_wilson(k, n, z=1.96):
    p = k / n
    den = 1 + z**2 / n
    centro = (p + z**2 / (2 * n)) / den
    mitad = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / den
    return centro - mitad, centro + mitad


def es_optimo(serie):
    return serie.str.contains('OPTIMO', case=False, na=False)


# ----------------------------------------------------------------------
# COMPONENTE VISUAL: SEMÁFORO DE 3 LUCES (1 ENCENDIDA, 2 APAGADAS)
# ----------------------------------------------------------------------
def mostrar_semaforo_visual(tiempo_promedio, tiempo_ideal=5.0, tolerancia=1.2):
    """
    Renderiza un semáforo visual con 3 luces utilizando HTML/CSS dentro de Streamlit.
    """
    umbral_amarillo = tiempo_ideal * tolerancia

    if tiempo_promedio <= tiempo_ideal:
        estado = "verde"
        titulo = "🟢 ESTADO ÓPTIMO"
        desc = f"El tiempo promedio (**{tiempo_promedio:.2f} min**) cumple perfectamente con la meta (≤ {tiempo_ideal:.1f} min)."
    elif tiempo_promedio <= umbral_amarillo:
        estado = "amarillo"
        titulo = "🟡 ESTADO DE PRECAUCIÓN"
        desc = f"El tiempo promedio (**{tiempo_promedio:.2f} min**) excede un poco la meta ideal, manteniéndose en zona de precaución."
    else:
        estado = "rojo"
        titulo = "🔴 ESTADO CRÍTICO (DEMORADO)"
        desc = f"El tiempo promedio (**{tiempo_promedio:.2f} min**) supera el límite aceptable. Se requiere optimizar procesos."

    c_rojo = "#ff4d4d" if estado == "rojo" else "#4d1a1a"
    op_rojo = "1" if estado == "rojo" else "0.3"

    c_amarillo = "#ffcc00" if estado == "amarillo" else "#4d4d00"
    op_amarillo = "1" if estado == "amarillo" else "0.3"

    c_verde = "#2ecc71" if estado == "verde" else "#143d21"
    op_verde = "1" if estado == "verde" else "0.3"

    html_semaforo = f"""
    <div style="background-color: #1e1e1e; padding: 20px; border-radius: 12px; border: 1px solid #333; margin-bottom: 20px; display: flex; align-items: center; justify-content: space-around;">
        <div style="text-align: center;">
            <div style="display: flex; gap: 12px; background: #111; padding: 10px 18px; border-radius: 30px; border: 2px solid #444; box-shadow: inset 0 0 10px rgba(0,0,0,0.8);">
                <div style="width: 30px; height: 30px; background-color: {c_rojo}; opacity: {op_rojo}; border-radius: 50%; box-shadow: 0 0 { '15px ' + c_rojo if estado=='rojo' else 'none'};"></div>
                <div style="width: 30px; height: 30px; background-color: {c_amarillo}; opacity: {op_amarillo}; border-radius: 50%; box-shadow: 0 0 { '15px ' + c_amarillo if estado=='amarillo' else 'none'};"></div>
                <div style="width: 30px; height: 30px; background-color: {c_verde}; opacity: {op_verde}; border-radius: 50%; box-shadow: 0 0 { '15px ' + c_verde if estado=='verde' else 'none'};"></div>
            </div>
        </div>
        <div style="flex-grow: 1; margin-left: 20px;">
            <h4 style="margin: 0 0 5px 0; color: #fff; font-size: 18px;">{titulo}</h4>
            <p style="margin: 0; color: #ccc; font-size: 14px;">{desc}</p>
        </div>
    </div>
    """
    st.markdown(html_semaforo, unsafe_allow_html=True)


# ----------------------------------------------------------------------
# PANTALLA 1: REGISTRO
# ----------------------------------------------------------------------
def vista_registro():
    st.title("Registrar Atención")

    with st.form("registro_sencillo", clear_on_submit=True):
        grupo = st.radio("¿Qué tipo de servicio fue?",
                         ["Grupo 1 (Copias/Impresiones)", "Grupo 2 (Trámites/Pasaporte)"])
        servicio = st.text_input("¿Qué servicio exacto fue? (Ej. Copia, Foto, Pasaporte)")
        sexo = st.radio("Sexo del cliente:", ["F", "M"], horizontal=True)
        tiempo = st.number_input("¿Cuántos minutos demoró en total?",
                                 min_value=0.1, value=1.0, step=0.5)
        satisfaccion = st.radio("¿El cliente se fue contento con el servicio?",
                                ["Sí, OPTIMO 😊", "NO, hubo molestias 😞"])
        enviar = st.form_submit_button("Guardar en el sistema ✔️", use_container_width=True)

    if enviar:
        try:
            guardar_registro(
                grupo=1 if "Grupo 1" in grupo else 2,
                servicio=servicio.strip() if servicio.strip() else "Varios",
                sexo=sexo,
                minutos=tiempo,
                satisfaccion='OPTIMO' if "Sí" in satisfaccion else 'REGULAR',
            )
            st.success("¡Listo! Registro guardado. El dashboard y el análisis se actualizan solos.")
        except PermissionError:
            st.error("No se pudo guardar: el Excel está abierto en otro programa. "
                     "Ciérralo e inténtalo de nuevo.")
        except Exception as e:
            st.error(f"No se pudo guardar el registro: {e}")

    st.divider()
    st.subheader("Últimos registros")
    mostrar_ultimos()


@st.fragment(run_every=REFRESCO)
def mostrar_ultimos():
    df = cargar_datos()
    st.caption(f"Total de clientes registrados: {len(df)}")
    
    # Creamos una copia para mostrar los minutos formateados como MM:SS en la tabla visual
    df_visual = df[['GRUPO DE SERVICIO', 'SERVICIO', 'Minutos', 'SEXO', 'SATISFACCION']].tail(5).iloc[::-1].copy()
    df_visual['TIEMPO (MM:SS)'] = df_visual['Minutos'].apply(_formatear_minutos_a_mmss)
    df_visual = df_visual[['GRUPO DE SERVICIO', 'SERVICIO', 'TIEMPO (MM:SS)', 'SEXO', 'SATISFACCION']]
    
    st.dataframe(df_visual, use_container_width=True)


# ----------------------------------------------------------------------
# PANTALLA 2: DASHBOARD DESCRIPTIVO (se actualiza solo)
# ----------------------------------------------------------------------
@st.fragment(run_every=REFRESCO)
def vista_dashboard():
    st.title("Resumen de tu negocio")
    df = cargar_datos()
    st.caption(f"🔄 Actualización automática cada {REFRESCO} s · "
               f"Última lectura: {datetime.now().strftime('%H:%M:%S')}")

    tiempo_promedio = df['Minutos'].mean()
    felices = int(es_optimo(df['SATISFACCION']).sum())
    porcentaje_felices = felices / len(df) * 100

    mostrar_semaforo_visual(tiempo_promedio=tiempo_promedio, tiempo_ideal=REF_TIEMPO)

    col1, col2, col3 = st.columns(3)
    col1.metric("Clientes atendidos", len(df))
    col2.metric("Tiempo Promedio General", f"{_formatear_minutos_a_mmss(tiempo_promedio)} min")
    col3.metric("Clientes Felices", f"{porcentaje_felices:.0f}%")
    st.divider()

    st.subheader("Comparación por Grupo de Servicio")
    df_grupos = df.groupby('GRUPO DE SERVICIO')['Minutos'].mean().reset_index()
    df_grupos['GRUPO DE SERVICIO'] = "Grupo " + df_grupos['GRUPO DE SERVICIO'].astype(int).astype(str)
    
    # Agregamos texto formateado para las barras del gráfico
    df_grupos['Tiempo_Texto'] = df_grupos['Minutos'].apply(_formatear_minutos_a_mmss)
    
    fig1 = px.bar(df_grupos, x='GRUPO DE SERVICIO', y='Minutos',
                  color='GRUPO DE SERVICIO', text='Tiempo_Texto')
    st.plotly_chart(fig1, use_container_width=True)


# ----------------------------------------------------------------------
# PANTALLA 3: INFERENCIA ESTADÍSTICA (se actualiza sola)
# ----------------------------------------------------------------------
@st.fragment(run_every=REFRESCO)
def vista_inferencial():
    st.title("Análisis Estadístico del Negocio")
    st.caption(f"🔄 Actualización automática cada {REFRESCO} s · "
               f"Última lectura: {datetime.now().strftime('%H:%M:%S')}")

    df = cargar_datos()
    n_total = len(df)

    if n_total < 2:
        st.warning("Se necesitan al menos 2 registros para hacer el análisis.")
        return

    # ---------------- OBJETIVO 1 ----------------
    st.subheader("1. Estimación General del Servicio")
    st.write("Confiabilidad estadística basada en las atenciones actuales:")

    mean_time = df['Minutos'].mean()
    std_time = df['Minutos'].std()
    if std_time > 0:
        ci_time = stats.t.interval(0.95, n_total - 1, loc=mean_time,
                                    scale=std_time / np.sqrt(n_total))
    else:
        ci_time = (mean_time, mean_time)

    felices = int(es_optimo(df['SATISFACCION']).sum())

    lo = stats.beta.ppf(0.025, felices, n_total - felices + 1) if felices > 0 else 0.0
    hi = stats.beta.ppf(0.975, felices + 1, n_total - felices) if felices < n_total else 1.0
    ci_prop = (lo * 100, hi * 100)

    t_inf_str = f"{_formatear_minutos_a_mmss(ci_time[0])} a {_formatear_minutos_a_mmss(ci_time[1])}"
    st.info(f"🕒 **Tiempo de atención:** Con un 95% de seguridad, el tiempo real promedio por cliente se encuentra entre **{t_inf_str} min**.")
    st.info(f"😊 **Satisfacción del cliente:** Con un 95% de seguridad, el porcentaje real de clientes satisfechos está entre **{ci_prop[0]:.2f}% y {ci_prop[1]:.2f}%**.")

    st.divider()

    # ---------------- OBJETIVO 2 ----------------
    st.subheader("2. Comparación entre Grupos de Servicio")
    st.write("Evaluando cuál grupo de servicio rinde mejor:")

    g1 = df[df['GRUPO DE SERVICIO'] == 1]
    g2 = df[df['GRUPO DE SERVICIO'] == 2]

    if len(g1) > 1 and len(g2) > 1:
        n1, n2 = len(g1), len(g2)
        v1, v2 = g1['Minutos'].var(), g2['Minutos'].var()

        diff_mean = g1['Minutos'].mean() - g2['Minutos'].mean()
        se_diff = np.sqrt(v1 / n1 + v2 / n2)
        if se_diff > 0:
            gl = se_diff**4 / ((v1 / n1)**2 / (n1 - 1) + (v2 / n2)**2 / (n2 - 1))
            tcrit = stats.t.ppf(0.975, gl)
            ci_m = (diff_mean - tcrit * se_diff, diff_mean + tcrit * se_diff)
        else:
            ci_m = (diff_mean, diff_mean)

        if ci_m[1] < 0:
            concl = "El **Grupo 1**, correspondiente a Servicios de atención rápida es significativamente más rápido."
        elif ci_m[0] > 0:
            concl = "El **Grupo 2** correspondiente a Servicios de gestión documental es significativamente más rápido."
        else:
            concl = "No hay una diferencia de velocidad importante entre ambos grupos."
        st.success(f"⚖️ **Diferencia de tiempos (G1 vs G2):** Entre {ci_m[0]:.2f} a {ci_m[1]:.2f} min. {concl}")

        k1 = int(es_optimo(g1['SATISFACCION']).sum())
        k2 = int(es_optimo(g2['SATISFACCION']).sum())
        p1, p2 = k1 / n1, k2 / n2
        l1, u1 = intervalo_wilson(k1, n1)
        l2, u2 = intervalo_wilson(k2, n2)
        dp = p1 - p2
        ci_p = (dp - np.sqrt((p1 - l1)**2 + (u2 - p2)**2),
                dp + np.sqrt((u1 - p1)**2 + (p2 - l2)**2))
        st.success(f"⚖️ **Diferencia de satisfacción entre Servicios de atención rápida y Servicios de gestión documental:** El margen de diferencia en clientes felices está entre **{ci_p[0] * 100:.1f}% y {ci_p[1] * 100:.1f}%**.")

        if min(n1, n2) < 10:
            st.warning(f"⚠️ Precaución: Un grupo tiene pocos datos ({min(n1, n2)} registros). Se recomienda registrar más atenciones para mayor precisión.")
    else:
        st.warning("Se necesitan al menos 2 datos en cada grupo para hacer comparaciones.")

    st.divider()

    # ---------------- OBJETIVO 3 ----------------
    st.subheader("3. Evaluación de Metas del Negocio")
    st.write(f"Verificación frente a las metas globales configuradas (**{REF_TIEMPO} min** de tiempo y **{REF_SAT}%** de satisfacción):")

    if std_time > 0:
        _, p_val = stats.ttest_1samp(df['Minutos'], REF_TIEMPO, alternative='less')
        if p_val < 0.05:
            st.success(f"🟢 **Meta de Tiempo Cumplida:** El tiempo promedio está por debajo de la meta de {REF_TIEMPO} min (Demostrado estadísticamente).")
        else:
            st.error(f"🔴 **Meta de Tiempo No Superada:** Todavía no hay evidencia suficiente de que estemos por debajo de los {REF_TIEMPO} min.")
    else:
        st.warning("No hay suficiente variación de datos para probar la media.")

    p_0 = REF_SAT / 100
    p_val_prop = stats.binomtest(felices, n_total, p_0, alternative='greater').pvalue
    if p_val_prop < 0.05:
        st.success(f"🟢 **Meta de Satisfacción Cumplida:** Superamos con éxito la meta del {REF_SAT}% de clientes felices.")
    else:
        st.error(f"🔴 **Meta de Satisfacción No Alcanzada:** Aún no alcanzamos consistentemente la meta del {REF_SAT}% de satisfacción.")


# ----------------------------------------------------------------------
# PANTALLA 4: TABLA (se actualiza sola)
# ----------------------------------------------------------------------
@st.fragment(run_every=REFRESCO)
def vista_excel():
    st.title("Base de datos en tiempo real")
    df = cargar_datos()
    st.caption(f"🔄 Actualización automática cada {REFRESCO} s · "
               f"{len(df)} registros · Última lectura: {datetime.now().strftime('%H:%M:%S')}")
    st.dataframe(df, use_container_width=True)

    with open(ARCHIVO, "rb") as f:
        st.download_button("⬇️ Descargar Excel", f.read(), file_name="Datos.xlsx",
                           mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")


# ----------------------------------------------------------------------
# ENRUTADOR
# ----------------------------------------------------------------------
if menu == "⏱️ Registrar un cliente":
    vista_registro()
elif menu == "📊 Dashboard Descriptivo":
    vista_dashboard()
elif menu == "📈 Análisis Inferencial":
    vista_inferencial()
else:
    vista_excel()
