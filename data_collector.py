#%%
import os
import openai
import sys
sys.path.append('../..')

#import panel as pn  # GUI
#pn.extension()

from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv()) # read local .env file

import datetime
current_date = datetime.datetime.now().date()
if current_date < datetime.date(2023, 9, 2):
    llm_name = "gpt-3.5-turbo-0301"
else:
    llm_name = "gpt-3.5-turbo"
print(llm_name)

from langchain_openai import ChatOpenAI

llm = ChatOpenAI(openai_api_key="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")

#%%
llm.invoke("how can langsmith help with testing?")

#import os
#os.environ["LANGCHAIN_TRACING_V2"] = "true"
#os.environ["LANGCHAIN_ENDPOINT"] = "https://api.langchain.plus"
#os.environ["LANGCHAIN_API_KEY"] = "..."

#%%
"""load docs"""
from langchain.document_loaders import PyPDFLoader

loader = PyPDFLoader("docs/41_Pirou_hydrogen extraction.pdf")
pages = loader.load()

#%%
"""split into chunks"""

#%%
"""convert to vectors"""

#%%
"""prompt"""

