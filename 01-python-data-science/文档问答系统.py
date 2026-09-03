# 1. 加载文档（把产品手册读进来）
from langchain_community.document_loaders import TextLoader
loader = TextLoader("product_manual.txt")
documents = loader.load()

# 2. 切成小段 + 存成向量 + 创建检索器（一口气干完）
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

texts = RecursiveCharacterTextSplitter(chunk_size=500).split_documents(documents)
vectorstore = Chroma.from_documents(texts, OpenAIEmbeddings())
retriever = vectorstore.as_retriever()

# 3. 创建 QA 链（把检索 + 模型串起来）
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI

qa_chain = RetrievalQA.from_chain_type(
    llm=ChatOpenAI(),
    retriever=retriever
)

# 4. 提问！
answer = qa_chain.invoke("产品保修期是多久？")
print(answer)