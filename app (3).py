"""CEMEX - Inferencia únicamente. No entrena modelos ni elimina registros."""
import io
import hashlib
import re
import unicodedata

import joblib
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Reclasificación CEMEX", page_icon="📄")
st.title("Reclasificación automática de costos CEMEX – By UDEM")
st.write("Carga el Excel mensual de SAP para obtener una propuesta de reclasificación sin volver a entrenar el modelo.")

FEATURES = {
    "Denom.clase de coste_num": "Denom.clase de coste",
    "Cl.coste_num": "Cl.coste",
    "Costo Estandar_num": "Costo Estandar",
    "Descrip.clases coste_num": "Descrip.clases coste",
}

ALIASES = {
    "Denom.clase de coste": ["Denom clase de coste", "Denom. clase de coste"],
    "Cl.coste": ["Cl coste", "Clase coste"],
    "Costo Estandar": ["Costo Estándar", "Costo estándar"],
    "Descrip.clases coste": ["Descrip. clases coste", "Descripción clases coste"],
}


def canonical(s):
    s = unicodedata.normalize("NFKD", str(s))
    s = "".join(c for c in s if not unicodedata.combining(c))
    return re.sub(r"[^a-z0-9]+", "", s.casefold())


def find_column(columns, desired):
    variants = {canonical(x) for x in [desired] + ALIASES.get(desired, [])}
    matches = [c for c in columns if canonical(c) in variants]
    if len(matches) != 1:
        raise ValueError(f"No se pudo identificar sin ambigüedad «{desired}». Encontradas: {matches}")
    return matches[0]


def load_dictionary(file_bytes):
    dictionary = pd.read_excel(io.BytesIO(file_bytes), dtype={"Número": "Int64"})
    required = {"Variable original", "Número", "Categoría"}
    if not required.issubset(dictionary.columns):
        raise ValueError("El diccionario debe tener las columnas: Variable original, Número y Categoría.")
    mappings = {}
    for variable, group in dictionary.groupby("Variable original", dropna=False):
        group = group.dropna(subset=["Número"])
        if group.empty:
            continue
        # La conversión usada en el notebook histórico fue astype(str) antes de LabelEncoder.
        keys = group["Categoría"].astype(str).tolist()
        vals = group["Número"].astype(int).tolist()
        if len(set(keys)) != len(keys) or len(set(vals)) != len(vals):
            raise ValueError(f"El diccionario tiene códigos duplicados en {variable}.")
        mappings[str(variable)] = dict(zip(keys, vals))
    return mappings


def prepare_X(original, maps):
    X = pd.DataFrame(index=original.index)
    known = pd.Series(True, index=original.index)
    for feature, variable in FEATURES.items():
        if variable not in maps:
            raise ValueError(f"Falta la codificación histórica de «{variable}» en Diccionario_transform.xlsx.")
        column = find_column(original.columns, variable)
        # Reproduce la representación de texto del LabelEncoder histórico.
        raw = original[column].astype(str)
        mapped = raw.map(maps[variable])
        known &= mapped.notna()
        X[feature] = mapped
    return X, known


def predict_original(original, model, maps):
    if "Reclasificación" in original.columns:
        raise ValueError("El Excel ya contiene «Reclasificación». Cámbiale el nombre antes de procesar para no sobrescribirlo.")
    if not hasattr(model, "predict") or not hasattr(model, "classes_"):
        raise ValueError("El archivo no contiene un modelo entrenado válido con predict() y classes_.")
    if "Categoría de costo" not in maps:
        raise ValueError("Falta la codificación de la variable objetivo «Categoría de costo» en el diccionario histórico.")
    X, known = prepare_X(original, maps)
    out = original.copy()
    out["Reclasificación"] = pd.Series(pd.NA, index=out.index, dtype="object")
    if known.any():
        pred_codes = model.predict(X.loc[known, list(FEATURES)].astype(int))
        inverse = {int(v): k for k, v in maps["Categoría de costo"].items()}
        predictions = pd.Series(pred_codes, index=out.index[known]).map(inverse)
        if predictions.isna().any():
            raise ValueError("El modelo devolvió códigos objetivo que no están en el diccionario histórico.")
        out.loc[known, "Reclasificación"] = predictions
    return out, int(known.sum()), int((~known).sum())


st.info("**CONFIDENCIALIDAD:** Herramienta de uso exclusivo para personal autorizado. No cargues datos de CEMEX en una aplicación pública ni compartas archivos o modelos sin autorización. Este aviso no reemplaza controles de acceso ni medidas de seguridad.")

with st.expander("Configuración del modelo histórico (una vez por sesión)", expanded=True):
    model_file = st.file_uploader("1. Sube modelo_cemex.joblib (modelo YA entrenado)", type=["joblib"], key="model")
    mapping_file = st.file_uploader("2. Sube Diccionario_transform.xlsx (codificaciones históricas)", type=["xlsx"], key="dictionary")

st.subheader("Sube tu Excel mensual original (.xlsx)")
monthly_file = st.file_uploader("3. Selecciona el Excel que quieres reclasificar", type=["xlsx"], key="monthly")

if monthly_file is not None:
    try:
        original = pd.read_excel(io.BytesIO(monthly_file.getvalue()), dtype=object)
        st.write(f"Cantidad de registros base original: **{len(original):,}**")
        if st.button("Reclasificar", type="primary", disabled=(model_file is None or mapping_file is None)):
            with st.spinner("Aplicando modelo guardado; no se está entrenando..."):
                # Los archivos joblib/pickle pueden ejecutar código al abrirlos: usar SOLO artefactos propios y de confianza.
                model = joblib.load(io.BytesIO(model_file.getvalue()))
                maps = load_dictionary(mapping_file.getvalue())
                result, n_pred, n_missing = predict_original(original, model, maps)
                buf = io.BytesIO()
                with pd.ExcelWriter(buf, engine="openpyxl", datetime_format="YYYY-MM-DD HH:MM:SS") as writer:
                    result.to_excel(writer, sheet_name="Base reclasificada", index=False)
                st.session_state["result_bytes"] = buf.getvalue()
                st.session_state["result_counts"] = (len(original), n_missing, n_pred)
                st.session_state["signature"] = tuple(
                    hashlib.sha256(f.getvalue()).hexdigest()
                    for f in (monthly_file, model_file, mapping_file)
                )  # Evitar descargas de una carga anterior.
    except Exception as exc:
        st.error(f"No se pudo procesar el archivo: {exc}")

    signature = st.session_state.get("signature")
    if (signature is not None and model_file is not None and mapping_file is not None
        and signature == tuple(hashlib.sha256(f.getvalue()).hexdigest()
                               for f in (monthly_file, model_file, mapping_file))):
        n_total, n_missing, n_pred = st.session_state["result_counts"]
        st.write(f"Cantidad de registros sin reclasificar: **{n_missing:,}**")
        st.write(f"Cantidad de registros reclasificados: **{n_pred:,}**")
        st.write("Archivo clasificado. Descarga el .xlsx")
        st.download_button(
            "Descargar base reclasificada .xlsx",
            data=st.session_state["result_bytes"],
            file_name="CEMEX_base_reclasificada.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            type="primary",
        )
    elif model_file is None or mapping_file is None:
        st.caption("Para activar Reclasificar, sube arriba el modelo entrenado y el diccionario histórico.")

st.caption("Se conservan todas las filas y columnas originales. Las categorías desconocidas para el modelo histórico quedan sin clasificar, nunca se inventa un código. Los resultados son predicciones que requieren validación.")
