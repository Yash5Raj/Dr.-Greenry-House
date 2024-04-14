from apikey import apikey_huggingface, apikey_openai
import os
from langchain.llms import HuggingFaceHub
from langchain.llms import OpenAI

os.environ['HUGGINGFACEHUB_API_TOKEN'] = apikey_huggingface
os.environ['OPENAI_API_KEY'] = apikey_openai
hf_llm = HuggingFaceHub(repo_id="google/flan-t5-xxl", model_kwargs={"temperature":0.9})
openai_llm = OpenAI(temperature=0.9)
