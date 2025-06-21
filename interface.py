import streamlit as st
import os
from motor import comparar_libros, graficar_tendencia
import pandas as pd

st.title("Analizador de textos literarios")
archivos = st.file_uploader("Sube tus archivos (PDF o TXT)", 
                            type=['pdf', 'txt'], 
                            accept_multiple_files=True)

def detectar_intencion(opcion_usuario):
    """Detecta la intención del usuario a partir de la opción seleccionada."""
    tipo_mapping = {
        "Análisis Gramatical": "gramatical",
        "Análisis de personajes": "personajes",
        "Ambos Análisis": "ambos"
    }
    return tipo_mapping.get(opcion_usuario, None)

rutas = []
if archivos:
    for archivo in archivos:
        with open(archivo.name, "wb") as f:
            f.write(archivo.getbuffer())
        rutas.append(archivo.name)

opcion = st.radio("Selecciona tipo de análisis:", 
                  ("Análisis Gramatical", "Análisis de personajes", "Ambos Análisis"), 
                  horizontal=True)

# Detectar intención
intencion = detectar_intencion(opcion)

if st.button("Analizar") and rutas:
    if not intencion:
        st.error("No se pudo detectar la intención del análisis.")
    else:
        with st.spinner("Analizando textos..."):
            try:
                resultados = comparar_libros(rutas, intencion)
            except Exception as e:
                st.error(f"Error durante el análisis: {e}")
                resultados = None
        if resultados:
            st.success("Análisis completado")
            # Mostrar gráficos según el tipo
            if intencion == 'ambos':
                fig_gram, fig_pers = graficar_tendencia(resultados, intencion, libros=resultados['nombres_libros'])
                st.subheader('Análisis Gramatical')
                st.pyplot(fig_gram)
                st.subheader('Análisis de personajes')
                st.pyplot(fig_pers)
            else:
                fig = graficar_tendencia(resultados, intencion, libros=resultados['nombres_libros'])
                st.pyplot(fig)
            # Mostrar datos tabulares para gramatical si aplica
            if intencion in ['gramatical', 'ambos']:
                st.subheader("Datos Gramaticales")
                gram_df = pd.DataFrame(resultados['gramatical'])
                gram_df.index = resultados['nombres_libros']
                st.dataframe(gram_df)
    # Limpiar archivos temporales
    for ruta in rutas:
        try:
            if os.path.exists(ruta):
                os.remove(ruta)
        except Exception as e:
            st.warning(f"No se pudo eliminar el archivo temporal {ruta}: {e}")
