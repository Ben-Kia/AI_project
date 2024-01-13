#%%
import os
import glob
import openai
import sys
sys.path.append('../..')

from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv()) # read local .env file

from langchain_openai import ChatOpenAI
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter, MarkdownHeaderTextSplitter, TokenTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain.llms import OpenAI
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain.chains.query_constructor.base import AttributeInfo
from langchain.retrievers import ContextualCompressionRetriever, SVMRetriever, TFIDFRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain.chains import RetrievalQA, ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory

from transformers import AutoTokenizer
import numpy as np


#%%
llm_name = "gpt-3.5-turbo"
#llm = ChatOpenAI(openai_api_key="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")
os.environ["OPENAI_API_KEY"] = "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"
llm = ChatOpenAI(model_name=llm_name, temperature=0)
llm.invoke("please repeat: I am alive and well.")


#%%
def check_credit_usage():
    try:
        # Make a request to the API
        account_info = openai.Account.retrieve()

        # Extract relevant information
        total_tokens_used = account_info['usage']['total_tokens']
        prompt_tokens_remaining = account_info['usage']['prompt_tokens_remaining']

        # Display the results
        print(f"Total tokens used: {total_tokens_used}")
        print(f"Prompt tokens remaining: {prompt_tokens_remaining}")

    except Exception as e:
        print(f"Error: {e}")


def pretty_print_docs(docs):
    print(f"\n{'-' * 100}\n".join([f"Document {i+1}:\n\n" + d.page_content for i, d in enumerate(docs)]))


#%% Load docs
"""loads any number of pdfs that is placed in the ./docs directory"""

pdf_list = glob.glob('./docs/*.pdf')
docs = []

for pdf in pdf_list:
    loader = PyPDFLoader(pdf)

    # loading into docs
    docs.extend(loader.load())

print(f"{len(pdf_list)} file(s) were loaded into {len(docs)} pages.")


#%% Split into chunks
splitter = RecursiveCharacterTextSplitter(
    chunk_size = 1500,
    chunk_overlap = 150,
    separators=["\n\n", "\n", "\. ", " ", ""]
)

doc_splits = splitter.split_documents(docs)
#text_splits = splitter.split_text(joined_page_text)

#%% Store in vectors and embed
"""here you can choose between OpenAI and HuggingFace embeddings"""

embedding_function = HuggingFaceBgeEmbeddings(
    model_name="BAAI/bge-base-en-v1.5",
    model_kwargs={'device': 'cuda'}, # works with CUDA, too, but requires CUDA-enabled torch
    encode_kwargs={'normalize_embeddings': True},
    query_instruction="Represent this sentence for searching relevant passages: "
)  

vectordb = Chroma.from_documents(
    doc_splits,
    embedding_function,
    persist_directory="./chroma_db"
)

print(vectordb._collection.count())

"""other retrievers are also available"""

#svm_retriever = SVMRetriever.from_texts(text_splits, embedding_function)
#tfidf_retriever = TFIDFRetriever.from_texts(text_splits)



#%% 
"""filters can be applied on metadata to limit the search
for example: fitler={"source:"path/to/source"} can be used to limit the search 
to one specific source"""

question = "Different combinations of sintering, reduction, and hydration were used to prepare electrolyte membranes. How long did the product of each preparation approach remained crack-free in atmospheric conditions?"

#%% Retrieval using a self-query retriever
"""uses an LLM to extract the query string and the metadata filter so you don't 
have to pass it manually"""

# do the following descriptions more specifically

metadata_field_info = [
    AttributeInfo(
        name="source",
        description="Each pdf file in the ./docs directory is a scientific paper", #The lecture the chunk is from, should be one of `docs/cs229_lectures/MachineLearning-Lecture01.pdf`, `docs/cs229_lectures/MachineLearning-Lecture02.pdf`, or `docs/cs229_lectures/MachineLearning-Lecture03.pdf`",
        type="string",
    ),
    AttributeInfo(
        name="page",
        description="The page from the paper",
        type="integer",
    ),
]

# We could use the GPT to summarize each pdf and write an individual description.
document_content_description = "paper on protonic ceramic cells" # do we need this argument? if so make it more specific.

llm_self_query_ret = OpenAI(model='gpt-3.5-turbo-instruct', temperature=0) # for factual answers

retriever = SelfQueryRetriever.from_llm(
    llm_self_query_ret,
    vectordb,
    document_content_description,
    metadata_field_info,
    verbose=True
)

retrieved_docs = retriever.get_relevant_documents(question)
pretty_print_docs(retrieved_docs)

#%%Contextual Compression

#%% Alternative retrievers

#%% RetrievalQA Chain

#%% Prompting

#Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer. Use three sentences maximum. Keep the answer as concise as possible. Always say "thanks for asking!" at the end of the answer. 
template = """Use the following pieces of context to answer the question at the end. If you don't know the answer, just say "UNKNOWN". 
{context}
Question: {question}
Helpful Answer:"""
QA_CHAIN_PROMPT = PromptTemplate.from_template(template)

memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True
)

qa_chain = RetrievalQA.from_chain_type(
    llm,
    retriever=retriever,#vectordb.as_retriever(),
    return_source_documents=True,
    chain_type_kwargs={"prompt": QA_CHAIN_PROMPT}
)

result = qa_chain({"query": question})
result["result"]

#%% Adding a memory component is required to have chat functionality, otherwise, it will be a one-shot Q&A
"""
memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True
)

qa_conversational = ConversationalRetrievalChain.from_llm(
    llm,
    retriever=vectordb.as_retriever(),
    memory=memory
)

result = qa_conversational({"question": question})
result['answer']
"""