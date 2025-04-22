from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer, util
import pdfplumber

app = FastAPI()

# Leer el PDF y preparar las preguntas y respuestas
pdf_path = "Promociones_Tratamientos_Estetica.pdf"

def leer_preguntas_respuestas(pdf_path):
    with pdfplumber.open(pdf_path) as pdf:
        text = ""
        for page in pdf.pages:
            text += page.extract_text() + "\n"

    preguntas_respuestas = []
    lineas = text.split("\n")
    pregunta_actual = ""
    respuesta_actual = ""

    for linea in lineas:
        if linea.startswith("P:"):
            if pregunta_actual and respuesta_actual:
                preguntas_respuestas.append({
                    "pregunta": pregunta_actual.strip(),
                    "respuesta": respuesta_actual.strip()
                })
                respuesta_actual = ""
            pregunta_actual = linea[2:].strip()
        elif linea.startswith("R:"):
            respuesta_actual += linea[2:].strip() + " "
        else:
            respuesta_actual += linea.strip() + " "
    if pregunta_actual and respuesta_actual:
        preguntas_respuestas.append({
            "pregunta": pregunta_actual.strip(),
            "respuesta": respuesta_actual.strip()
        })
    return preguntas_respuestas

# Preparar el modelo y los embeddings
modelo = SentenceTransformer('all-MiniLM-L6-v2')
preguntas_respuestas = leer_preguntas_respuestas(pdf_path)
lista_preguntas = [item['pregunta'] for item in preguntas_respuestas]
embeddings_preguntas = modelo.encode(lista_preguntas, convert_to_tensor=True)

# Esquema de la petición
class PreguntaUsuario(BaseModel):
    pregunta: str

@app.post("/preguntar")
def preguntar(data: PreguntaUsuario):
    pregunta_usuario = data.pregunta
    embedding_usuario = modelo.encode(pregunta_usuario, convert_to_tensor=True)
    similitudes = util.cos_sim(embedding_usuario, embeddings_preguntas)
    indice_mejor = similitudes.argmax().item()
    respuesta = preguntas_respuestas[indice_mejor]['respuesta']
    pregunta_encontrada = preguntas_respuestas[indice_mejor]['pregunta']
    return {
        "pregunta_usuario": pregunta_usuario,
        "pregunta_encontrada": pregunta_encontrada,
        "respuesta": respuesta
    }
