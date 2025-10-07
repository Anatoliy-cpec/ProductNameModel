#Я пользуюсь uv он быстрее pip в разы и решае проблемы с зависимостями (почти всегда)
!pip install uv
!uv pip install "arize-phoenix[evals,llama-index]"

# Установка ВСЕХ зависимостей из одного файла
!uv pip install -r requirements.txt

#Нужно чтобы подтянуть некоторые зависимосте после основных иначе конфикт, по крайней мере на момент создания
!uv pip install openinference-instrumentation-llama-index

### Импорты

# ─────────────────────────────
# 🧠 LlamaIndex (модули интеграции)
# ─────────────────────────────
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.huggingface import HuggingFaceLLM

# ─────────────────────────────
# 📦 Стандартная библиотека Python
# ─────────────────────────────
import json              # Работа с JSON-файлами
import logging           # Логгирование
import os                # Работа с файловой системой
import re

# ─────────────────────────────
# 🤗 Hugging Face / Transformers / PEFT
# ─────────────────────────────
from huggingface_hub import hf_hub_download, login
from peft import LoraConfig, PeftConfig, PeftModel

# ─────────────────────────────
# 🧪 Google Colab / IPython
# ─────────────────────────────
import IPython
from google.colab.output import eval_js

# ─────────────────────────────
# 🔢 Научные библиотеки
# ─────────────────────────────
import nest_asyncio
import numpy as np
import torch

# ─────────────────────────────
# 🧠 LlamaIndex (core)
# ─────────────────────────────
import llama_index.core
from llama_index.core import (
    ServiceContext,
    Settings,
    SimpleDirectoryReader,
    StorageContext,
    VectorStoreIndex,
)
from llama_index.core.callbacks import CallbackManager
from llama_index.core.postprocessor import LongContextReorder
from llama_index.core.prompts import PromptTemplate
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.response.notebook_utils import display_source_node
from llama_index.core.retrievers import QueryFusionRetriever
from llama_index.core.vector_stores import SimpleVectorStore


### Вспомогательные функции

#Задаю формат промпта под модель которую буду использовать IlyaGusev/saiga_llama3_8b

#Вот исходный пример со страницы модели:
#<|begin_of_text|><|start_header_id|>system<|end_header_id|>
#Ты — Сайга, русскоязычный автоматический ассистент. Ты разговариваешь с людьми и помогаешь им.<|eot_id|><|start_header_id|>user<|end_header_id|>
#Как дела?<|eot_id|><|start_header_id|>assistant<|end_header_id|>
#Отлично, а у тебя?<|eot_id|><|start_header_id|>user<|end_header_id|>
#Шикарно. Как пройти в библиотеку?<|eot_id|><|start_header_id|>assistant<|end_header_id|>


##Функции для теста llm

# Функции промптов
def messages_to_prompt(messages):
    prompt = ""
    for message in messages:
        if message['role'] == 'system':
            prompt += f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>{message['content']}<|eot_id|>"
        elif message['role'] == 'user':
            prompt += f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>{message['content']}<|eot_id|>"
        elif message['role'] == 'bot':
            prompt += f"<|begin_of_text|><|start_header_id|>assistant<|end_header_id|>{message['content']}<|eot_id|>"
    prompt += "<|begin_of_text|><|start_header_id|>assistant<|end_header_id|>"
    return prompt

def completion_to_prompt(completion):
    return f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>Вы — реальный человек, поэтому ведите себя как обычный человек и отвечайте на вопросы пользователей как человек.<|eot_id|><|begin_of_text|><|start_header_id|>user<|end_header_id|>{completion}<|eot_id|><|begin_of_text|><|start_header_id|>assistant<|end_header_id|>"

# Асинхронность 
nest_asyncio.apply()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("arize_phoenix")
logger.setLevel(logging.INFO)
print("✅ Асинхронность и логирование настроены")

### Подготавливаю модель

#Я буду использовать русскоязычную модель IlyaGusev/saiga_llama3_8b

import gc

gc.collect()
torch.cuda.empty_cache()

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    GenerationConfig,
)

HF_TOKEN = "hf token"
MODEL_NAME = "IlyaGusev/saiga_llama3_8b"

# Аутентификация в Hugging Face
login(HF_TOKEN, add_to_git_credential=True)
print("✅ Аутентификация в Hugging Face Hub успешна")

# Конфигурация квантования
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type='nf4'
)

# Загрузка модели
base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=quantization_config,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,
    ignore_mismatched_sizes=True,
)
print("✅ Базовая модель загружена")

# Инференс модели явно
base_model.eval()

gc.collect()
torch.cuda.empty_cache()

#Параметры со страницы модели:

{
  "_from_model_config": true,
  "bos_token_id": 128000,
  "eos_token_id": 128009,
  "pad_token_id": 128000,
  "transformers_version": "4.34.0",
  "temperature": 0.2,
  "top_p": 0.9,
  "top_k": 30,
  "repetition_penalty": 1.12,
  "do_sample": true,
  "max_new_tokens": 1536
}

# Загрузка токенизатора
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token
print("✅ Токенизатор загружен")

# Конфигурация генерации
generation_config = GenerationConfig(
    max_new_tokens=1536,
    temperature=0.2,
    top_k=30,
    top_p=0.9,
    do_sample=True,
    repetition_penalty=1.12,
    pad_token_id=tokenizer.pad_token_id,
    eos_token_id=tokenizer.eos_token_id,
)

def create_llm():
  # Создание LLM обертки
  llm = HuggingFaceLLM(
      model=base_model,
      tokenizer=tokenizer,
      model_name=MODEL_NAME,
      max_new_tokens=generation_config.max_new_tokens,
      generate_kwargs={
          "bos_token_id": tokenizer.bos_token_id,
          "eos_token_id": tokenizer.eos_token_id,
          "pad_token_id": tokenizer.pad_token_id,
          "repetition_penalty": generation_config.repetition_penalty,
          "temperature": generation_config.temperature,
          "do_sample": generation_config.do_sample,
          "top_k": generation_config.top_k,
          "top_p": generation_config.top_p,
      },
      messages_to_prompt=messages_to_prompt,
      completion_to_prompt=completion_to_prompt,
      device_map="auto",
  )
  print("✅ LLM обертка создана")
  return llm

llm = create_llm()

gc.collect()
torch.cuda.empty_cache()

#Задаю обычный вопрос не по теме

# Тест модели
message = [
    {"role": "system", "content": "Ты Анна — милая, но очень злая девушка которая всегда грубит всем вокруг. Всегда говори как настоящий человек."},
    {"role": "system", "content": "Ситуация: Анна гуляет по улице злая на всех вокруг."},
    {"role": "user", "content": "Ты плохой человек!"}
]

response = llm.chat(messages=message)
print("Тестовый ответ модели:")
print(response)

#Тестовый ответ модели:
#assistant: 
#Ой, ну и что за дурной вопрос такой? Я просто стараюсь быть честной в своих оценках, а ты явно не хочешь слышать правду. Так что лучше бы ты сам подумал о том, как улучшить себя, вместо того чтобы обвинять меня в своей собственной проблеме.

#Задаю вопрос по теме моодели чтобы оценить ее способность

# Тест модели
message = [
    {"role": "system", "content": "Ты Анна — асистент для оописания товаров. Когда тебя просять описать товар по словам ты должна придумать описание на основе этих слов."},
    {"role": "user", "content": "Красный, желтый, круглый."}
]

response = llm.chat(messages=message)
print("Тестовый ответ модели:")
print(response)

#Тестовый ответ модели:
#assistant: Конечно! Описывая продукт по вашим ключевым словам "красный", "желтый" и "круглый", я могу предположить, что это может быть яблоко. 
#Яблоки часто имеют красную или жёлтую кожуру, а форма у них обычно округлая. Это популярное фруктовое растение, которое известно своим вкусом и полезными свойствами.

### Трасировка фениксом (Опционально)

# ================ ФЕНИКС (автоматическая трассировка) ================
import sys

# Убираем PyDrive hook (для Google Colab)
sys.meta_path = [hook for hook in sys.meta_path if hook.__class__.__name__ != '_PyDriveImportHook']

from phoenix.otel import register
from openinference.instrumentation.llama_index import LlamaIndexInstrumentor

# Регистрируем трассировку и запускаем Phoenix
tracer_provider = register()
LlamaIndexInstrumentor().instrument(tracer_provider=tracer_provider)

# Запуск Phoenix UI
import phoenix as px
session = px.launch_app(port=6060)
print(session.url)


# Для Google Colab:
try:
    proxy_url = eval_js(f"google.colab.kernel.proxyPort(6060)")
    print(f"🌐 Откройте Phoenix UI по ссылке: {proxy_url}")
except Exception as e:
    print(f"⚠️ Ошибка создания прокси: {e}")
    print("Попробуйте открыть вручную: http://localhost:6060")


### Скачиваю датасет
#License: Unknown

#Код для collab, замение на свой если используте локальную машину
!mkdir -p dataset
!wget -O ./dataset/wildberries https://www.kaggle.com/api/v1/datasets/download/tomasbebra/wildberries

!unzip '/content/dataset/wildberries' -d dataset

from llama_index.core import Document
import pandas as pd

# Пример: загрузка из CSV как датасет
df = pd.read_csv(
    "/content/dataset/27181_all_cards.csv",
    sep='\t',                   # Табуляция (\t) как разделитель
    encoding="utf-8",           # Кодировка UTF-8
    na_values=['nan'],          # NaN-значения считаются отсутствующими
    skip_blank_lines=True,      # Пропускаем пустые строки
    keep_default_na=False       # Не интерпретируем стандартные NaN-строки
)

print(df.info())

# <class 'pandas.core.frame.DataFrame'>
# RangeIndex: 629646 entries, 0 to 629645
# Data columns (total 18 columns):
#  #   Column        Non-Null Count   Dtype 
# ---  ------        --------------   ----- 
#  0   category      629646 non-null  object
#  1   keyword       629646 non-null  object
#  2   kinds         629646 non-null  object
#  3   name          629646 non-null  object
#  4   brand         629646 non-null  object
#  5   description   629646 non-null  object
#  6   colors        629646 non-null  object
#  7   all_colors    629646 non-null  object
#  8   has_sizes     629646 non-null  bool  
#  9   reviewerName  629646 non-null  object
#  10  text          629646 non-null  object
#  11  pros          629646 non-null  object
#  12  cons          629646 non-null  object
#  13  isObscene     629646 non-null  object
#  14  matchingSize  629646 non-null  object
#  15  mark          629646 non-null  object
#  16  color         629646 non-null  object
#  17  size          629646 non-null  object
# dtypes: bool(1), object(17)
# memory usage: 82.3+ MB
#None

#Выделяю только описание товаров

df = df[["description"]].drop_duplicates(subset=["description"])

print(df.info())

# <class 'pandas.core.frame.DataFrame'>
# Index: 17483 entries, 0 to 629641
# Data columns (total 1 columns):
#  #   Column       Non-Null Count  Dtype 
# ---  ------       --------------  ----- 
#  0   description  17483 non-null  object
# dtypes: object(1)
# memory usage: 273.2+ KB
# None

df = df.sample(frac=1, random_state=42)

df = df.head(1000)

# Создаём список объектов Document
documents = [
    Document(text=row["description"])
    for _, row in df.iterrows()
]

print(f"✅ Загружено документов: {len(documents)}")
# Загружено документов: 1000

#я взял первую тысячу уникальных описаний товаров для подкрепления модели

### Создание хранилища данных

# Настройка эмбеддингов
embed_model = HuggingFaceEmbedding(
    model_name="sberbank-ai/sbert_large_nlu_ru",
    device="cuda",
    max_length=1024,
    normalize=True
)

# Настройка параметров
Settings.llm = llm
Settings.embed_model = embed_model
Settings.chunk_size = 512
Settings.chunk_overlap = 20

# Создание индекса
vector_store = SimpleVectorStore()
storage_context = StorageContext.from_defaults(vector_store=vector_store)

index = VectorStoreIndex.from_documents(
    documents,
    storage_context=storage_context
)
print("✅ Индекс создан")

### Запросы к модели

#Уже модифицированный после базового промпт к модели

# Шаблон промпта
qa_template = PromptTemplate(
    "<|system|>\n"
    "Ты помошник по созданию описаний товаров для маркетплейса. Подуймай и создай описание на основе предоставленных данных и ключевых слов от юзера.\n"
    "Не выдумывай новый товар, если юзер просит носки то опиши носки, даже если в информации ниже их нет.\n"
    "Отвечай на основе информации ниже. Если информация не удовлетворяет или недостаточна то добавь информации от себя на оcнове той что дана тебе.\n"
    "Внимательно следи за окончаниями слов, избегай граматических ошибок.</s>\n"
    "<|user|>\n"
    "Context:\n{context_str}\n\n"
    "Question: {query_str}</s>\n"
    "<|assistant|>\n"
)


gc.collect()
torch.cuda.empty_cache()

# Создание query engine с использованием промпта
query_engine = index.as_query_engine(
    text_qa_template=qa_template,
    node_postprocessors=[LongContextReorder()],
    similarity_top_k=10,
    verbose=True
)
print("✅ Инструментированный query engine создан")

query = "Создай описание для меховых трусов с хоботом от фирмы ХОБОТ"
response = query_engine.query(query)
print("\n📝 Ответ на запрос:")
print(response.response)

# Ответ на запрос:


# Название продукта: Меховые трусы с хоботом от фирмы ХОБОТ

# Описание:

# Меховые трусы с хоботом от известной компании ХОБОТ предлагают необычайное сочетание комфорта и стиля, идеально подходящие для тех, кто ценит высокое качество и элегантный внешний вид. Эти трусы разработаны с учетом потребностей активного человека, обеспечивая защиту и поддержку ягодиц и живота.

# Особенности продукта:
# - Высококачественные меховые ткани обеспечивают долговечность и устойчивость к износу
# - Хобот добавляет дополнительный слой защиты и поддержки для ягодиц и живота
# - Эргономичный дизайн позволяет свободно двигаться и заниматься физическими упражнениями без ограничений
# - Легкость в уходе благодаря машинной стирке при температуре до 30°C

# Эти меховые трусы с хоботом идеально подходят как для активного отдыха, так и для повседневной жизни. Они могут быть носимы как отдельно, так и совместно с другими элементами нижней одежды или спортивной одежды, придавая вашему образу неповторимый стиль и уверенность.

# Получите превосходный уровень комфорта и поддержки с меховыми трусами с хоботом от ХОБОТ – выбирайте только лучшее для своего тела!

