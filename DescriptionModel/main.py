import re
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GRU, Dense, Embedding, Dropout, Input
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
import pickle
import gc

# Очистка GPU и сессий
tf.config.list_physical_devices('GPU')
gc.collect()
tf.keras.backend.clear_session()

# Загрузка данных
df = pd.read_csv('./wildberries/cards.csv', sep='\t', low_memory=False)

df = df[['name','description', 'keyword', 'text']]

df = df.groupby('keyword').head(300)

# Удаляем строки с пропущенными значениями в ключевых столбцах
df.dropna(inplace=True)

def preprocess_text(text):
    text = text.lower()  # Приведение к нижнему регистру
    text = re.sub(r'[^\w\s]', '', text)  # Удаление знаков препинания
    return text


input = df.copy()

input = input.apply(lambda row: f"{row['description']} {row['text']} {row['keyword']}", axis=1)

input = input.apply(preprocess_text)

df['name'] = df['name'].apply(preprocess_text)

# X и y
X = input.values
y = df['name'].values




# Преобразуем тексты в последовательности
tokenizer = Tokenizer(num_words=10000, oov_token="<OOV>")
tokenizer.fit_on_texts(np.concatenate((X, y), axis=0))  # Совмещение X и y для токенизации

# Добавим токены <START> и <END>
start_index = len(tokenizer.word_index) + 1
end_index = start_index + 1

tokenizer.word_index["<START>"] = start_index
tokenizer.word_index["<END>"] = end_index
tokenizer.index_word[start_index] = "<START>"
tokenizer.index_word[end_index] = "<END>"

X_seq = tokenizer.texts_to_sequences(X)
y_seq = tokenizer.texts_to_sequences(y)

max_len_input = max(len(seq) for seq in X_seq)
max_len_output = max(len(seq) for seq in y_seq)

X_padded = pad_sequences(X_seq, maxlen=max_len_input, padding='post')
y_padded = pad_sequences(y_seq, maxlen=max_len_output, padding='post')

# Разделяем данные на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X_padded, y_padded, test_size=0.2, random_state=55)

# Сохраняем размер словаря
vocab_size = len(tokenizer.word_index) + 1

# Построение модели с функциональным API
# Вход для кодировщика
encoder_input = Input(shape=(max_len_input,))
encoder_embedding = Embedding(vocab_size, 254)(encoder_input)
encoder_gru, state_h = GRU(128, return_state=True)(encoder_embedding)

# Вход для декодера
decoder_input = Input(shape=(max_len_output - 1,))
decoder_embedding = Embedding(vocab_size, 254)(decoder_input)
decoder_gru = GRU(128, return_sequences=True)(decoder_embedding, initial_state=[state_h])
decoder_dropout = Dropout(0.2)(decoder_gru)
decoder_output = Dense(vocab_size, activation='softmax')(decoder_dropout)

# Модель
model = Model(inputs=[encoder_input, decoder_input], outputs=decoder_output)

model.compile(loss='sparse_categorical_crossentropy', optimizer=Adam(learning_rate=0.01), metrics=['accuracy'])

# Подготовка данных для обучения
decoder_input_seq_train = y_train[:, :-1]
decoder_target_seq_train = y_train[:, 1:]

decoder_input_seq_test = y_test[:, :-1]
decoder_target_seq_test = y_test[:, 1:]

# Обучение модели
history = model.fit(
    [X_train, decoder_input_seq_train],  # Входные данные
    decoder_target_seq_train,            # Целевые данные
    validation_data=(
        [X_test, decoder_input_seq_test], 
        decoder_target_seq_test
    ),
    batch_size=64,
    epochs=2
)

# Сохраняем модель и токенайзер
with open('tokenizer.pickle', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

model.save('descriptor.h5')

# Функция генерации названия
def generate_name(input_text, tokenizer, model, max_len_input, max_len_output):
    input_seq = tokenizer.texts_to_sequences([input_text])
    input_padded = pad_sequences(input_seq, maxlen=max_len_input, padding='post')

    decoder_input_seq = np.zeros((1, 1))  # Начальный ввод для декодера с токеном <START>
    decoder_input_seq[0, 0] = tokenizer.word_index["<START>"]
    generated = []

    for _ in range(max_len_output):
        predictions = model.predict([input_padded, decoder_input_seq])
        predicted_id = np.argmax(predictions[0, -1, :])

        if predicted_id == tokenizer.word_index["<END>"]:  # Конец генерации
            break

        word = tokenizer.index_word.get(predicted_id, "<UNK>")
        generated.append(word)

        # Обновляем decoder_input_seq
        decoder_input_seq = np.zeros((1, len(generated)))  # Обновляем вход для декодера
        decoder_input_seq[0, -1] = predicted_id

    return " ".join(generated)

# # Пример использования:
sample_input = "блузка хлопковая белая для женщин"
generated_name = generate_name(sample_input, tokenizer, model, max_len_input, 10)
print("Сгенерированное название:", generated_name)
