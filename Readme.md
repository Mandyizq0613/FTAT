# 📚 Analizador Literario - FTAT

**Herramienta para análisis profundo de textos literarios en español**


## 🔍 Descripción

Este proyecto permite analizar textos literarios en español para extraer:
- Distribución gramatical (adjetivos, verbos, sustantivos, adverbios)
- Evolución de personajes (detección y seguimiento de muertes)
- Comparativa entre múltiples obras

## ✨ Características principales

- ✅ Soporte para archivos PDF y TXT
- ✅ Detección automática de capítulos
- ✅ Análisis gramatical con spaCy
- 🎭 Identificación de personajes con NER
- 💀 Sistema avanzado de detección de muertes
- 📊 Visualización comparativa
- 📈 Gráficos profesionales con Seaborn/Matplotlib

## 📦 Dependencias

```bash
pip install PyPDF2 spacy pandas matplotlib seaborn numpy
python -m spacy download es_core_news_sm