# Worklog

## LangChain

**LLM Adapters** - `langchain.llms.OpenAI`, `langchain.llms.HuggingFaceHub`, и т.д. Это адаптеры, которые отвечают за обрабатывание http запросов к API, сериализацию/десериализацию, управлением параметрами (температура, токены и т.д.).
Адаптеры не зависят от архитектуры модели, а только от API интерфеса.

### Conceptions

**Chat Messages**:
System - Helpful background context that tell the AI what to do
Human - Messages that are intended to represent the user
AI - Messages that show what the AI responded with

```python
chat(
    [
        SystemMessage(content="You are a nice AI bot that helps a user figure out where to travel in one short sentence"),
        HumanMessage(content="I like the beaches where should I go?"),
        AIMessage(content="You should go to Nice, France"),
        HumanMessage(content="What else should I do when I'm there?")
    ]
)
```

**Document**

```python
Document(
  page_content="This is my document. It is full of text that I've gathered from other places",
  metadata={
    'my_document_id' : 234234,
    'my_document_source' : "The LangChain Papers",
    'my_document_create_time' : 1680013019
  }
)
```

#### Models

**Language Model** - модель обрабатывающая текст (одно сообщение)
**Chat Model** - модель, работающая с последовательностью сообщений
**Function Calling Models** - то же, что и чат, но отдает структурированный формат, например json
**Text Embedding Model** - переводит текст в вектор, чтобы можно было сравнить его с другими текстами

#### Prompts

**Prompts** - инструкция, для модели

**Prompt Template** - объект-помошник, который комбинирует пользовательский input с фиксированным шаблоном

```python
template= """
I really want to travel to {location}. What should I do there?

Respond in one short sentence
"""

prompt = PromptTemplate(
    input_variables=["location"],
    template=template,
)

final_prompt = prompt.format(location='Rome')
```

**Examples** - примеры входящих-выходящих сообщений

### Outputs

Работать с выводом можно двумя способами:

1. Сообщать модели, какой вывод хочешь получить, и парсить его.
2. Использовать OpenAI Functions - это рекомендуемый метод, он позволяет использовать предтренированные модели выдавать результат по заданному типу (например pydantic).

```python
from langchain.pydantic_v1 import BaseModel, Field
from typing import Optional

class Person(BaseModel):
    name: str = Field(..., description="The person's name")
    age: int = Field(..., description="The person's age")
    fav_food: Optional[str] = Field(None, description="The person's favorite food")

from langchain.chains.openai_functions import create_structured_output_chain

llm = ChatOpenAI(model='gpt-4-0613', openai_api_key=openai_api_key)

chain = create_structured_output_chain(Person, llm, prompt)
chain.run("Sally is 13, Joey just turned 12 and loves spinach. Caroline is 10 years older than Sally.")
```

```python
import enum

llm = ChatOpenAI(model='gpt-4-0613', openai_api_key=openai_api_key)

class Product(str, enum.Enum):
    CRM = "CRM"
    VIDEO_EDITING = "VIDEO_EDITING"
    HARDWARE = "HARDWARE"

class Products(BaseModel):
    products: Sequence[Product] = Field(..., description="The products mentioned in a text")

chain = create_structured_output_chain(Products, llm, prompt)
chain.run("The CRM in this demo is great. Love the hardware. The microphone is also cool. Love the video editing")
```

### Indexes

Индексация

#### Document Loaders

Простой способ импортировать данные из других источником. У `langchain_community` есть огромный список готовых загрузчиков.

```python
from langchain.document_loaders import HNLoader
loader = HNLoader("https://news.ycombinator.com/item?id=88005553535")
data = loader.load()
print (f"Found {len(data)} comments")
print (f"Here's a sample:\n\n{''.join([x.page_content[:150] for x in data[:2]])}")
```

#### Text Splitter

Разделитель длинных материалов на чанки.

```python
text_splitter = RecursiveCharacterTextSplitter(
    # Set a really small chunk size, just to show.
    chunk_size = 150,
    chunk_overlap  = 20,
)

texts = text_splitter.create_documents([pg_work])
```

#### Retrievers

Простой способ обогатить модель информацией из документов.

```python
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings

loader = TextLoader('data/PaulGrahamEssays/worked.txt')
documents = loader.load()
# Get your splitter ready
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50)

# Split your docs into texts
texts = text_splitter.split_documents(documents)

# Get embedding engine ready
embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)

# Embed your texts
db = FAISS.from_documents(texts, embeddings)
# Init your retriever. Asking for just 1 document back
retriever = db.as_retriever()
```

```python
docs = retriever.get_relevant_documents("what types of things did the author want to build?")
print("\n\n".join([x.page_content[:200] for x in docs[:2]]))
```

#### Vectors Stores

Базы данных для хранения векторов, ембеддингов.

```python
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings

loader = TextLoader('data/PaulGrahamEssays/worked.txt')
documents = loader.load()

# Get your splitter ready
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50)

# Split your docs into texts
texts = text_splitter.split_documents(documents)

# Get embedding engine ready
embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
print (f"You have {len(texts)} documents")
```

#### Memory

Память, помогает модели запоминать информацию. LangChain дает много разных реализаций памяти.

Пример с ChatMessageHistory:

```python
from langchain.memory import ChatMessageHistory
from langchain.chat_models import ChatOpenAI

chat = ChatOpenAI(temperature=0, openai_api_key=openai_api_key)

history = ChatMessageHistory()

history.add_ai_message("hi!")

history.add_user_message("what is the capital of france?")
history.messages
```

### Chains

Совмещает разные LLM вызовы и действия.

**Simple Sequential Chains** - простая цепочка где вы можете использовать output LLM как вход в следующий вызов. Хорошо для разрыва задач (и сфокусирована на LLM)

```python
from langchain.llms import OpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.chains import SimpleSequentialChain

llm = OpenAI(temperature=1, openai_api_key=openai_api_key)
template = """Your job is to come up with a classic dish from the area that the users suggests.
% USER LOCATION
{user_location}

YOUR RESPONSE:
"""
prompt_template = PromptTemplate(input_variables=["user_location"], template=template)

# Holds my 'location' chain
location_chain = LLMChain(llm=llm, prompt=prompt_template)
template = """Given a meal, give a short and simple recipe on how to make that dish at home.
% MEAL
{user_meal}

YOUR RESPONSE:
"""
prompt_template = PromptTemplate(input_variables=["user_meal"], template=template)

# Holds my 'meal' chain
meal_chain = LLMChain(llm=llm, prompt=prompt_template)
overall_chain = SimpleSequentialChain(chains=[location_chain, meal_chain], verbose=True)
review = overall_chain.run("Rome")
```

**Summraization Chain** - простой способ пройтись по документам и выдать суммаризацию.

```python
from langchain.chains.summarize import load_summarize_chain
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

loader = TextLoader('data/PaulGrahamEssays/disc.txt')
documents = loader.load()

# Get your splitter ready
text_splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=50)

# Split your docs into texts
texts = text_splitter.split_documents(documents)

# There is a lot of complexity hidden in this one line. I encourage you to check out the video above for more detail
chain = load_summarize_chain(llm, chain_type="map_reduce", verbose=True)
chain.run(texts)

```

### Agents

> [!qote] Official Docs:
> Некоторые приложения потребуют не только предопределенной цепочки вызовов для LLMS/других инструментов, но и для неизвестной цепочки, которая зависит от ввода пользователя. В этих типах цепочек есть «агент», который имеет доступ к набору инструментов. В зависимости от ввода пользователя, агент может решить, какие, если таковые имеются, из этих инструментов для вызова.

В основном вы используете LLM не только для вывода текста, но и для принятия решений. Крутость и сила этой ф-ции не может быть переоценена.
Сэм Альтман подчеркивает, что LLM - это хороший «двигатель рассуждений». Агент пользуется этим преимуществом.

**Agents** - Языковая модель, которая стимулирует принятие решений.

Более конкретно, агент получает вход и возвращает ответ, соответствующий действию, чтобы принять вместе с action input. Подробнее о типах агентов [here](https://python.langchain.com/api_reference/langchain/agents/langchain.agents.agent_types.AgentType.html).

> [!note] вместо агентов рекомендуется использовать langgraph.

**Tools** - 'возможности' агентов. Это абстракция поверх фц-ий которые упрощают для LLMs (и agents) взаимодействие с другими инструментами (напримре google search).

**Toolkit** - коллекция инструментов.

```python
from langchain.agents import load_tools
from langchain.agents import initialize_agent
from langchain.llms import OpenAI
import json

llm = OpenAI(temperature=0, openai_api_key=openai_api_key)
serpapi_api_key=os.getenv("SERP_API_KEY", "YourAPIKey")
toolkit = load_tools(["serpapi"], llm=llm, serpapi_api_key=serpapi_api_key)
agent = initialize_agent(toolkit, llm, agent="zero-shot-react-description", verbose=True, return_intermediate_steps=True)
response = agent({"input":"what was the first album of the band that Natalie Bergman is a part of?"})
```

### Extraction

> [!note] Kor - либа для экстракции данных из текста.

**Response Schema** - схема реагирования делает две вещи:

- автогенерация подсказки с инструкциями в формате bonafide.
- десериализует выход от llm

### Evaluation

**Evaluation** - процесс оценки эффективности и анализ производительности приложений с LLM. Тустирование ответов на предопределенных критериев илли эталовон, что бы удебиться, что она соответствует требуемым стандартам качетсва и выполняет поставленные задачи.

LangSmith помогает с этим:

- облегчает создание и курирование наборов данных через трассировки и аннотационных функций
- предоставляет структуру оценки, которая помогате определить метрики и запустить приложение с вашим набором данных
- позволяет отслеживать результаты с течением времени и автоматически запускать ваших оценщиков по расписанию или как часть CI/код

## Infrastructure

# Benchmarks

## RAG101 Triage Benchmark

### Tasks

**so_vector** - используется дамп вопросов со stackoverflow, это 2 миллиона вопросов прошедших через embedding `multi-qa-mpnet-base-cos-v1`

**dense_vector** - используется набор данных yandex DEEP1B, это 10M векторов 96 размерности.

**openai_vector** - pending

# Templates

## Design Document for \<PROJECT\>

### Simple First

```
Introduction (golas of document, terms, how data will flow through the system, etc.)
System Overview
- High level Description ()
- Technology Stack ()
Technical Approach
- Tools (what tools will be used)
System Architecture
- High level Architecture (how data will flow through the system, focus on architecture, not implementation)
- Deployment (how will the system be deployed)
- SubSystem Design (how divided up is the system and what include in each subsystem)
Class Diagrams (by subsystems)
- Schedule Subsystem Diagrams (uml for subsystems)
- Application Flow (how data through the system, where tools are used, ...)
Sequence Diagram (by subsystems)
```

### Simple Second

```
Introduction
- Purpose (application goal, what include in design document)
- Scope (major components of application)
- References

System Overview

System Components
- Descomposition Description
- Dependency Description
- Interface Description
- UX/UI

Detailed Design
- Module Detailed Design
- Data Detailed Design
```

## References

1. [MCPLearn / github.com](https://github.com/microsoft/lets-learn-mcp-python#quick-start0)
2. [AIResearchHub / MCP Learn / github.com](https://github.com/microsoft/lets-learn-mcp-python/blob/main/AIResearchHub/server.py)
3. [LangChain Cookbook / github.com](https://github.com/gkamradt/langchain-tutorials/blob/main/LangChain%20Cookbook%20Part%201%20-%20Fundamentals.ipynb)
4. [LangSmithHub / smith.langchain.com](https://smith.langchain.com/hub)
5. []()
