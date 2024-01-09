#%%
import os
import openai
import sys
sys.path.append('../..')

#import panel as pn  # GUI
#pn.extension()

from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv()) # read local .env file

openai.api_key  = os.environ['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']

import datetime
current_date = datetime.datetime.now().date()
if current_date < datetime.date(2023, 9, 2):
    llm_name = "gpt-3.5-turbo-0301"
else:
    llm_name = "gpt-3.5-turbo"
print(llm_name)

from langchain_openai import ChatOpenAI

llm = ChatOpenAI(openai_api_key="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")


llm.invoke("how can langsmith help with testing?")


# %%


from openai import OpenAI
client = OpenAI()

completion = client.chat.completions.create(
  model="gpt-3.5-turbo",
  messages=[
    {"role": "system", "content": "You are a poetic assistant, skilled in explaining complex programming concepts with creative flair."},
    {"role": "user", "content": "Compose a poem that explains the concept of recursion in programming."}
  ]
)

print(completion.choices[0].message)
print("hi")
#%%