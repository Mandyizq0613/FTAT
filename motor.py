import re
import PyPDF2
import spacy
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os 
import numpy as np
from matplotlib.ticker import MaxNLocator
#carga el contenido del archivo devolviéndolo como texto
def cargar_archivos(ruta):
    if ruta.endswith('.pdf'):
        with open(ruta, 'rb') as archivo:
            lector = PyPDF2.PdfReader(archivo)
            texto = ""
            for pagina in range(len(lector.pages)):
                texto+= lector.pages[pagina].extract_text()
            return texto
    elif ruta.endswith('.txt'):
        with open(ruta, 'r' , encoding= 'utf-8') as archivo:
            return archivo.read()
    else:
        raise ValueError("Formato de archivo no soportado. Usa PDF o TXT, por favor :)")

# limpiar el texto de caracteres no deseados (purificar el texto)
def limpiar_texto(texto):
    texto = texto.lower()
    # Eliminar caracteres especiales pero conservar letras, espacios y puntuación básica
    texto = re.sub(r'[^a-záéíóúüñ .,;:!?¿¡\-\'"\n]', '', texto)
    # Eliminar múltiples espacios
    texto = re.sub(r'\s+', ' ', texto)
    return texto.strip()

# ahora dividimos el texto por capítulos (usando patrones comunes de separación)
def dividir_por_capítulos(texto):
    # Solo patrones claros de capítulos
    patrones = [
        r'\n+cap[ií]tulo\s+\d+\b',   # Capítulo 1, Capítulo 23
        r'\n+cap[ií]tulo\s+[ivxlc]+\b', # Capítulo I, Capítulo IV
        r'\n+chapter\s+\d+\b',         # Chapter 1 (por si hay libros en inglés)
        r'\n+chapter\s+[ivxlc]+\b'      # Chapter I
    ]
    patron_compilado = re.compile('|'.join(patrones), re.IGNORECASE)
    capitulos = patron_compilado.split(texto)
    # Si la primera división es muy corta (portada, índice), la descartamos
    if capitulos and len(capitulos[0].strip()) < 100:
        capitulos = capitulos[1:]
    return [cap.strip() for cap in capitulos if cap.strip()] #filtrar los elementos vacíos

# analizará el texto y contará las categorías gramaticales
def analizar_gramática(texto):
    nlp = spacy.load("es_core_news_sm")
    nlp.max_length = max(len(texto) + 1000, 1_000_000)
    # Procesar por capítulos si el texto es muy largo
    if len(texto) > 200_000:
        capitulos = dividir_por_capítulos(texto)
        conteo_total = {'adjetivos': 0, 'adverbios': 0, 'verbos': 0, 'sustantivos': 0, 'total_de_palabras': 0}
        for cap in capitulos:
            doc = nlp(cap)
            for token in doc:
                if not token.is_punct and not token.is_space:
                    conteo_total['total_de_palabras'] += 1
                    if token.pos_ == 'ADJ':
                        conteo_total['adjetivos'] += 1
                    elif token.pos_ == 'ADV':
                        conteo_total['adverbios'] += 1
                    elif token.pos_ == 'VERB':
                        conteo_total['verbos'] += 1
                    elif token.pos_ == 'NOUN':
                        conteo_total['sustantivos'] += 1
        for categoría in ['adjetivos', 'adverbios', 'verbos', 'sustantivos']:
            if conteo_total['total_de_palabras'] > 0:
                porcentaje = (conteo_total[categoría] / conteo_total['total_de_palabras']) * 100
                conteo_total[f'%_{categoría}'] = round(porcentaje, 2)
        return conteo_total
    else:
        doc = nlp(texto)
        conteo = {'adjetivos': 0 , 'adverbios': 0 , 'verbos': 0 , 'sustantivos': 0 , 'total_de_palabras': 0}
        for token in doc:
            if not token.is_punct and not token.is_space:
                conteo['total_de_palabras'] += 1
                if token.pos_ == 'ADJ':
                    conteo['adjetivos'] += 1
                elif token.pos_ == 'ADV':
                    conteo['adverbios'] += 1
                elif token.pos_ == 'VERB':
                    conteo['verbos'] += 1
                elif token.pos_ == 'NOUN':
                    conteo['sustantivos'] += 1
        for categoría in ['adjetivos', 'adverbios', 'verbos', 'sustantivos']:
            if conteo['total_de_palabras'] > 0:
                porcentaje = (conteo[categoría] / conteo['total_de_palabras']) * 100
                conteo[f'%_{categoría}'] = round(porcentaje, 2)
        return conteo

# detectar a los personajes (usando NER), devuelve una lista de los personajes
def detectar_personajes(texto):
    nlp = spacy.load("es_core_news_sm")
    nlp.max_length = max(len(texto) + 1000, 1_000_000)
    # Procesar por capítulos si el texto es muy largo
    if len(texto) > 200_000:
        capitulos = dividir_por_capítulos(texto)
        personajes = set()
        for cap in capitulos:
            doc = nlp(cap)
            for ent in doc.ents:
                if ent.label_ == "PER":
                    nombre = ' '.join(palabra.capitalize() for palabra in ent.text.split())
                    personajes.add(nombre)
        return sorted(personajes)
    else:
        doc = nlp(texto)
        personajes = set()
        for ent in doc.ents:
            if ent.label_ == "PER":
                nombre = ' '.join(palabra.capitalize() for palabra in ent.text.split())
                personajes.add(nombre)
        return sorted(personajes)

#identifica las muertes de personajes en cada capítulo
def identificar_muertes(capitulos):
    import spacy
    nlp = spacy.load("es_core_news_sm")
    nlp.max_length = 1_500_000
    verbos_muerte = {"morir", "fallecer", "perecer", "expirar", "desaparecer", "desvanecerse", "asesinar", "ejecutar", "matar", "suicidar", "ahorcar", "envenenar", "fusilar", "ahogar", "aplastar", "degollar", "decapitar", "apunalear", "aplastar", "aniquilar"}
    sustantivos_muerte = {"muerte", "fallecimiento", "asesinato", "desaparición", "ejecución", "suicidio", "homicidio", "tragedia", "funeral", "cadáver", "cuerpo", "restos"}
    expresiones_indirectas = [
        "se desvaneció de pronto", "no volvió a despertar", "dejó de respirar", "no volvió a ser visto", "no volvió a aparecer", "se apagó su luz", "se fue para siempre", "no regresó jamás", "no volvió a levantarse", "no volvió a moverse"
    ]
    muertes_por_capitulo = {}
    personajes_muertos = set()
    for i, capitulo in enumerate(capitulos):
        num_capitulo = i + 1
        muertes_por_capitulo[num_capitulo] = []
        if not capitulo.strip():
            continue
        doc = nlp(capitulo)
        personajes_cap = set([ent.text for ent in doc.ents if ent.label_ == "PER"])
        for sent in doc.sents:
            sent_text = sent.text.lower()
            # 1. Verbo de muerte con personaje como sujeto
            for token in sent:
                if token.lemma_ in verbos_muerte and token.pos_ == "VERB":
                    sujeto = [w for w in token.lefts if w.dep_ in ("nsubj", "nsubj:pass")]
                    for s in sujeto:
                        nombre = s.text.strip()
                        if nombre in personajes_cap and nombre not in personajes_muertos:
                            muertes_por_capitulo[num_capitulo].append(nombre)
                            personajes_muertos.add(nombre)
            # 2. Sustantivo de muerte con personaje como complemento
            for token in sent:
                if token.lemma_ in sustantivos_muerte and token.pos_ == "NOUN":
                    # Buscar complemento directo o posesivo
                    for child in token.children:
                        if child.dep_ in ("obj", "nmod", "poss"):
                            nombre = child.text.strip()
                            if nombre in personajes_cap and nombre not in personajes_muertos:
                                muertes_por_capitulo[num_capitulo].append(nombre)
                                personajes_muertos.add(nombre)
            # 3. Expresiones indirectas
            for expr in expresiones_indirectas:
                if expr in sent_text:
                    # Buscar sujeto de la oración
                    for token in sent:
                        if token.dep_ in ("nsubj", "nsubj:pass"):
                            nombre = token.text.strip()
                            if nombre in personajes_cap and nombre not in personajes_muertos:
                                muertes_por_capitulo[num_capitulo].append(nombre)
                                personajes_muertos.add(nombre)
        muertes_por_capitulo[num_capitulo] = list(set(muertes_por_capitulo[num_capitulo]))
    muertes_por_capitulo = {k: v for k, v in muertes_por_capitulo.items() if v}
    return muertes_por_capitulo

# cuenta la cantidad de muertes que hubo por capítulo
def contar_muertes_por_capítulo(muertes_por_capitulo, personajes_totales):
    estadisticas = {}
    vivos = set(personajes_totales)
    muertes_acumuladas = 0
    for cap in sorted(muertes_por_capitulo.keys()):
        muertes_cap = muertes_por_capitulo[cap]
        vivos -= set(muertes_cap)
        muertes_acumuladas += len(muertes_cap)
        estadisticas[cap] = {'muertes': len(muertes_cap) , 'vivos': len(vivos) , 'muertes_acumuladas': muertes_acumuladas , 'personajes_vivos': list(vivos)}
    return estadisticas

def graficar_tendencia(datos, tipo, libros=None):
    """
    Genera figuras matplotlib para visualización según el tipo de análisis
    
    Argumentos:
        datos: Resultados del análisis
        tipo: 'gramatical', 'personajes' o 'ambos'
        libros: Nombres de los libros
        
    Return:
        tuple: (fig_gram, fig_pers) o figura individual según el tipo
    """
    fig_gram = None
    fig_pers = None
    
    # ===== PARTE GRAMATICAL =====
    if tipo in ['gramatical', 'ambos']:
        # 1. Preparamos datos para el gráfico
        datos_grafico = []
        for i, libro_data in enumerate(datos['gramatical']):
            for cat in ['adjetivos', 'adverbios', 'verbos', 'sustantivos']:
                porcentaje = libro_data.get(f'%_{cat}', 0)
                datos_grafico.append({
                    'Libro': libros[i] if libros else f'Libro {i+1}',
                    'Categoría': cat.capitalize(),
                    'Porcentaje': porcentaje
                })
        
        # 2. Creamos DataFrame
        df = pd.DataFrame(datos_grafico)
        
        # 3. Creamos figura matplotlib
        fig_gram, ax = plt.subplots(figsize=(10, 6))
        
        # 4. Gráfico de barras
        libros_unicos = df['Libro'].unique()
        categorias = df['Categoría'].unique()
        
        # Configuración de posición de barras
        bar_width = 0.2
        index = np.arange(len(categorias))
        
        for i, libro in enumerate(libros_unicos):
            valores = df[df['Libro'] == libro]['Porcentaje']
            ax.bar(index + i * bar_width, valores, bar_width, label=libro)
        
        # 5. Personalización
        ax.set_xlabel('Categoría Gramatical')
        ax.set_ylabel('Porcentaje del total de palabras')
        ax.set_title('Distribución Gramatical (%)')
        ax.set_xticks(index + bar_width * (len(libros_unicos) - 1) / 2)
        ax.set_xticklabels(categorias)
        ax.legend()
        plt.tight_layout()
    
    # ===== PARTE PERSONAJES =====
    if tipo in ['personajes', 'ambos']:
        # 6. Creamos figura para personajes
        fig_pers, ax = plt.subplots(figsize=(10, 6))
        
        # 7. Procesamos cada libro
        for i, libro_data in enumerate(datos['personajes']):
            libro_nombre = libros[i] if libros else f'Libro {i+1}'
            
            # 8. Extraemos datos
            capitulos = sorted(libro_data.keys())
            muertes_acum = [libro_data[cap]['muertes_acumuladas'] for cap in capitulos]
            
            # 9. Gráfico de línea
            ax.plot(capitulos, muertes_acum, marker='o', label=libro_nombre)
            
            # 10. Añadimos puntos de muerte por capítulo
            for cap, stats in libro_data.items():
                if stats['muertes'] > 0:
                    ax.scatter(cap, stats['muertes_acumuladas'], 
                              s=100, alpha=0.7, edgecolors='black')
        
        # 11. Personalización
        ax.set_xlabel('Capítulo')
        ax.set_ylabel('Muertes acumuladas')
        ax.set_title('Evolución de Muertes de Personajes')
        ax.legend()
        ax.grid(True)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        plt.tight_layout()
    
    # 12. Devolvemos las figuras según el tipo de análisis
    if tipo == 'gramatical':
        return fig_gram
    elif tipo == 'personajes':
        return fig_pers
    else:  # 'ambos'
        return (fig_gram, fig_pers)

#comparar los lirbos según el tipo de análisis seleccionado
def comparar_libros(libros_rutas, opcion):
    resultados = {'gramatical':[] , 'personajes': [] , 'nombres_libros': []}
    for ruta in libros_rutas:
        texto = cargar_archivos(ruta)
        texto_limpio = limpiar_texto(texto)
        nombre = os.path.basename(ruta).split('.')[0]
        resultados['nombres_libros'].append(nombre)
        if opcion in ['gramatical' , 'ambos']:
            gramatica = analizar_gramática(texto_limpio)
            resultados['gramatical'].append(gramatica)
        if opcion in ['personajes', 'ambos']:
            capitulos = dividir_por_capítulos(texto)
            personajes = detectar_personajes(texto)
            muertes = identificar_muertes (capitulos)
            estadisticas = contar_muertes_por_capítulo(muertes , personajes)
            resultados['personajes'].append(estadisticas)
    return resultados