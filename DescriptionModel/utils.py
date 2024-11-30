import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences

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
