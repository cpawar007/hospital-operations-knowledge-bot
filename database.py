from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv

load_dotenv()

# Load documents
admission = TextLoader("ingestion/admission_process.txt").load()
discharge = TextLoader("ingestion/discharge_process.txt").load()
doctor_map = TextLoader("ingestion/doctor_mapping.txt").load()
emergency = TextLoader("ingestion/emergency_protocols.txt").load()
first_aid = TextLoader("ingestion/first_aid_guide.txt").load()
overview = TextLoader("ingestion/hospital_overview.txt").load()
policies = TextLoader("ingestion/hospital_policies.txt").load()
workflow = TextLoader("ingestion/hospital_workflow.txt").load()
hospital = TextLoader("ingestion/hospital.txt").load()

# Combine ALL docs (NO duplication)
all_docs = admission + discharge + doctor_map + emergency + first_aid + overview + policies + workflow + hospital

# Split into chunks
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1500,
    chunk_overlap=100
)

chunks = splitter.split_documents(all_docs)

# Embeddings
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Vector DB
vectorstore = FAISS.from_documents(
    chunks,
    embedding_model
)

vectorstore.save_local("Database_db")

print("DB created successfully")