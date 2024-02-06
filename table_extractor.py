#%%
import os
import shutil
import glob
import openai
import sys
import re
sys.path.append('../..')
from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv()) # read local .env file
from langchain_openai import ChatOpenAI
from langchain.llms import OpenAI
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain.chains.query_constructor.base import AttributeInfo
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
import csv


def setup_llm(api_key):
    llm_name = "gpt-3.5-turbo"
    os.environ["OPENAI_API_KEY"] = api_key
    llm = ChatOpenAI(model_name=llm_name, verbose=True, temperature=0)
    llm.invoke("please repeat: I am alive and well.")
    return llm

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


def load_pdfs(path_to_pdf=None):
    if not path_to_pdf:
        """loads any number of pdfs that is placed in the ./docs directory"""
        pdf_list = glob.glob('./docs/*.pdf')
    else:
        pdf_list = [path_to_pdf]
    
    docs = []
    for pdf in pdf_list:
        loader = PyPDFLoader(pdf)
        # loading into docs
        docs.extend(loader.load())
    
    print(f"{len(pdf_list)} file(s) loaded into {len(docs)} pages.")

    return docs


def split_docs(docs, chunk_size = 1500, chunk_overlap = 150, separators=["\n\n", "\n", "\. ", " ", ""]):
    """Split inputs into chunks"""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=separators
    )

    doc_splits = splitter.split_documents(docs)
    return doc_splits


def store(embedding, splits, device, persist_directory="./chroma_db", purge=True):
    """Store embeddings into vectors. Here you can choose between OpenAI and HuggingFace embeddings"""
    
    if embedding == 'openai':
        embedding_function = OpenAIEmbeddings()

    if embedding == 'huggingface':
        embedding_function = HuggingFaceBgeEmbeddings(
            model_name="BAAI/bge-base-en-v1.5",
            model_kwargs={'device': device},
            encode_kwargs={'normalize_embeddings': True},
            query_instruction="Represent this sentence for searching relevant passages: "
        )

    if purge:
        try:
            shutil.rmtree(persist_directory)
            print(f"Directory '{persist_directory}' and its contents successfully removed.")
            
        except OSError as e:
            print(f"Error: {e}")

    vectordb = Chroma.from_documents(
        splits,
        embedding_function, persist_directory=persist_directory
    )

    print(f"{vectordb._collection.count()} embeddings stored")
    return vectordb


def setup_retriever(search_type, k=5, fetch_k=3):
    
    if search_type == "selfquery":
        metadata_field_info = [
        AttributeInfo(
            name="source",
            description="The filename corresponding to the paper this chunk is from",
            type="string",
        ),
        AttributeInfo(
            name="page",
            description="The page from the paper",
            type="integer",
        ),
        ]

        # We could use the GPT to summarize each pdf and write an individual description.
        document_content_description = "Excerpts from a scientific paper"

        llm_self_query_ret = OpenAI(model='gpt-3.5-turbo-instruct', temperature=0)
            
        retriever = SelfQueryRetriever.from_llm(
            llm_self_query_ret,
            vectordb,
            document_content_description,
            metadata_field_info,
            verbose=True
        )

    if search_type == "similarity":
        retriever = vectordb.as_retriever(search_type="similarity", search_kwargs={"k": k})
    
    if search_type == "mmr":
        retriever = vectordb.as_retriever(search_type="mmr", search_kwargs={"k": k, "fetch_k":fetch_k})

    return retriever

def prompt(question, template, method, retriever=None):
    
    if method == "stuff":
        """
        llm_chain = LLMChain(llm=llm, prompt=prompt)
        
        chain = StuffDocumentsChain(
            llm_chain=llm_chain,
            document_prompt=document_prompt,
            document_variable_name=document_variable_name
        )

        prompt_result = chain({"query": question})
        """

    if method == "qa":
        QA_CHAIN_PROMPT = PromptTemplate.from_template(template)

        qa_chain = RetrievalQA.from_chain_type(
            llm,
            retriever=retriever,
            return_source_documents=True,
            verbose=True,
            chain_type_kwargs={"prompt": QA_CHAIN_PROMPT}
        )

        prompt_result = qa_chain({"query": question})
    return prompt_result


def create_filename(input_string, max_length=100):
    valid_chars = '-_.() abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
    processed_string = ''.join(c for c in input_string if c in valid_chars)
    processed_string = processed_string.replace(' ', '_')
    processed_string = processed_string.replace('.', '_')
    processed_string = re.sub('_+', '_', processed_string)
    processed_string = processed_string.strip('_')
    max_length = max_length
    processed_string = processed_string[:max_length]

    return processed_string


def save2csv(data_string, filename):
    lines = data_string.split('\n')
    with open(f'{filename}.csv', 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        
        # Write header
        header = lines[0].split(',')
        csv_writer.writerow(header)
        
        # Write data
        for line in lines[1:]:
            row = line.split(',')
            csv_writer.writerow(row)

    print(f"CSV file saved as '{filename}.csv'")

api_key = input("please provide the api key or leave blank for the default key: ", ).strip() or "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"
llm = setup_llm(api_key)

path_to_pdf = input("please provide a path to the pdf file or put it in the ./docs directory: ")
docs = load_pdfs(path_to_pdf) #'./docs/backup_docs/43_Chen_hydration induced chemical expansion.pdf'

page_no = -1 + int(input("please specify the page where the table is located in: "))

#purge the chroma directory
shutil.rmtree('C:\codestuff\AI_project\chroma_db')
print("chroma_db directory purged.")


#%%
doc_splits = split_docs([docs[page_no]], 10000, 250)

vectordb = store('huggingface', doc_splits, device='cuda', purge=True)

#%% Prompting

table_title = str(input("please specify the title of the table: "))
#table_title = "Table 2 e Electrochemical performance of symmetrical PCC cells"

#question = f"Rewrite the information in \"{table_title}\" from the document into csv format."
question = f"Rewrite the contents of \"{table_title}\" into csv format."

# "What parameters affected electrolyte membrane cracking in this paper?"
# "How long did the elecrolyte membranes produced by different preparations stay crack-free?"
# "What does the table titled \"Table 2 e Electrochemical performance of symmetrical PCC cells.\" contain?"


#prompt_template = """Use the following pieces of context to answer the question at the end. You are provided with scientific papers. You are not allowed to ignore any word, symbol, or number in the data. If you don't know the answer, just return UNKNOWN.
#{context}
#Question: {question}
#Helpful Answer:"""

prompt_template = """You are provided with an excerpt from a scientific paper as context.
This piece of context contains a table.
Rewrite the information in the specified table into csv format.
First try to identify the table header (there might be sub-headers),
Then identify table columns.
You are not allowed to ignore any word, symbol, or number in the data.
If you don't know the answer, just return UNKNOWN.
{context}
Question: {question}
Helpful Answer:"""

retriever = setup_retriever("mmr", 3, 1) #"selfquery", ("similarity", 3, 1) or ("mmr", 3, 1)

result = prompt(question, prompt_template, "qa", retriever)

if result["result"]!= "UNKNOWN":
    save2csv(result["result"], create_filename(table_title, 100))

# %%
