from tensorflow.keras.models import load_model
from utils import generate_name
import pickle

# Загрузка модели
loaded_model = load_model('descriptor.h5')
print("Модель загружена.")

# Загрузка токенизатора
with open('tokenizer.pickle', 'rb') as file:
    loaded_tokenizer = pickle.load(file)
print("Токенизатор загружен.")


# Генерация текста
input_example = "блузка женская"

# def generate_name(input_text, tokenizer, model, max_len_input, max_len_output):

generated_text = generate_name(input_example, loaded_tokenizer, loaded_model, 819, 10)
print(f"Сгенерированное название: {generated_text}")