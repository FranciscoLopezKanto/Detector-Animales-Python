from transformers import ViTFeatureExtractor, ViTForImageClassification
from PIL import Image
import torch
import io
from googletrans import Translator

translator = Translator()

MODEL_NAME = "google/vit-base-patch16-224"
feature_extractor = ViTFeatureExtractor.from_pretrained(MODEL_NAME)
model = ViTForImageClassification.from_pretrained(MODEL_NAME)

ANIMALES_CONOCIDOS = [
    "dog", "cat", "puppy", "kitten", "labrador", "retriever",
    "horse", "lion", "tiger", "elephant", "bear", "zebra", "giraffe",
    "monkey", "bird", "fish", "cow", "sheep", "goat", "pig", "hamster",
    "rabbit", "parrot", "chicken", "duck", "goose", "wolf", "fox",
    "leopard", "cheetah", "kangaroo", "panda", "camel", "deer", "otter",
    "raccoon", "penguin", "seal", "dolphin", "whale", "shark", "eagle",
    "falcon", "hawk", "owl", "bat", "squirrel", "donkey", "mule",
    "frog", "toad", "vulture", "snake", "lizard", "crocodile", "turtle"
]

CATEGORIAS = {
    "mamífero": ["dog", "cat", "horse", "lion", "tiger", "elephant", "bear", "zebra", "giraffe", "monkey", "cow", "sheep", "goat", "pig", "hamster", "rabbit", "wolf", "fox", "leopard", "cheetah", "kangaroo", "panda", "camel", "deer", "otter", "raccoon", "bat", "squirrel", "donkey", "mule"],
    "ave": ["bird", "parrot", "chicken", "duck", "goose", "penguin", "eagle", "falcon", "hawk", "owl", "vulture"],
    "pez": ["fish", "shark", "whale", "dolphin"],
    "reptil": ["snake", "lizard", "crocodile", "turtle"],
    "anfibio": ["frog", "toad", "salamander"]
}

def obtener_categoria(label: str) -> str:
    label_lower = label.lower()
    for categoria, keywords in CATEGORIAS.items():
        if any(word in label_lower for word in keywords):
            return categoria
    return None

def traducir_lista(texto: str) -> str:
    if not texto:
        return None
    partes = [p.strip() for p in texto.split(",")]
    traducciones = [translator.translate(p, src="en", dest="es").text for p in partes]
    return ", ".join(traducciones)

def predict_animal(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    inputs = feature_extractor(images=image, return_tensors="pt")
    outputs = model(**inputs)
    logits = outputs.logits
    predicted_class_idx = logits.argmax(-1).item()
    label = model.config.id2label[predicted_class_idx]
    confidence = torch.nn.functional.softmax(logits, dim=-1)[0][predicted_class_idx].item()

    # Verificar si cualquier palabra del label está en animales conocidos
    es_animal_detectado = any(word in label.lower() for word in ANIMALES_CONOCIDOS)

    if es_animal_detectado:
        es_animal = True
        animal_final = label
        animal_parecido = None
    else:
        es_animal = True  # Lo tratamos igual como animal
        animal_final = label
        animal_parecido = traducir_lista(label)

    categoria = obtener_categoria(label)

    return {
        "es_animal": es_animal,
        "animal_detectado": traducir_lista(animal_final),
        "categoria": categoria,
        "confianza": round(confidence, 2),
        "animal_parecido": animal_parecido
    }
