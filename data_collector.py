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
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceBgeEmbeddings
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
        docs = []

        for pdf in pdf_list:
            loader = PyPDFLoader(pdf)
            # loading into docs
            docs.extend(loader.load())

    else:
        loader = PyPDFLoader(path_to_pdf)
        docs.extend(loader.load())
    
    print(f"{len(pdf_list)} file(s) loaded into {len(docs)} pages.")

    return docs


def split_docs(docs):
    """Split inputs into chunks"""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size = 1500,
        chunk_overlap = 150,
        separators=["\n\n", "\n", "\. ", " ", ""]
    )

    doc_splits = splitter.split_documents(docs)
    return doc_splits


def store(embedding, splits, device):
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

    vectordb = Chroma.from_documents(
        splits,
        embedding_function#, persist_directory="./chroma_db"
    )

    print(f"{vectordb._collection.count()} embeddings stored")
    return vectordb


def chroma_retriever(k):
    retriever = vectordb.as_retriever(search_type="similarity", search_kwargs={"k": k})
    return retriever

def prompt(question, template, retriever, k):
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


llm = setup_llm("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")
docs = load_pdfs(path_to_pdf=None)
doc_splits = split_docs(docs)
vectordb = store('huggingface', doc_splits, device='cuda')

#%% Prompting


table_title = "Table 2 e Electrochemical performance of symmetrical PCC cells"

question = f"Rewrite the information in \"{table_title}\" from the document into csv format."

# "What parameters affected electrolyte membrane cracking in this paper?"
# "How long did the elecrolyte membranes produced by different preparations stay crack-free?"
# "What does the table titled \"Table 2 e Electrochemical performance of symmetrical PCC cells.\" contain?"

#Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer. Use three sentences maximum. Keep the answer as concise as possible. Always say "thanks for asking!" at the end of the answer. 
prompt_template = """Use the following pieces of context to answer the question at the end. You are provided with some scientific papers. If you don't know the answer, just say UNKNOWN. 
{context}
Question: {question}
Helpful Answer:"""

retriever = chroma_retriever(5)

result = prompt(question, prompt_template, retriever, 5)

save2csv(result["result"], f"{table_title}")

# %%
