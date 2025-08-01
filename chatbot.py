import os
from dotenv import load_dotenv
from langchain_upstage import UpstageEmbeddings
from langchain_community.document_loaders import Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore
from langchain import hub
from langchain_upstage import ChatUpstage
from langchain_core.messages import HumanMessage
from langchain.chains import RetrievalQA
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

def get_ai_answer(query):
    load_dotenv()

    upstage_api_key = os.environ.get("UPSTAGE_API_KEY")
    embeddings = UpstageEmbeddings(
        api_key=upstage_api_key,
        model="embedding-query"        
    )
    index_name = 'tax-table-index'
    #파인콘 벡터DB설정
    pinecone_api_key = os.environ.get("PINECONE_API_KEY")

    pc = Pinecone(api_key=pinecone_api_key)

    database = PineconeVectorStore.from_documents(document_list, embeddings, index_name=index_name)

    retriever = database.as_retriever(search_kwargs={'k': 2})
    retriever.invoke(query)

    prompt = hub.pull("rlm/rag-prompt")

    upstage_api_key = os.environ.get("UPSTAGE_API_KEY")
    #llm 모델 설정
    llm = ChatUpstage(
        api_key=upstage_api_key,
        model="solar-pro2"
    )
    #리트리버 chainType 설정
    qa_chain = RetrievalQA.from_chain_type(
        llm, 
        retriever=retriever,
        chain_type_kwargs={"prompt": prompt}
    )

    dictionary = ["사람을 나타내는 표현 -> 거주자"]
    prompt = ChatPromptTemplate.from_template(f"""
        사용자의 질문을 보고, 우리의 사전을 참고해서 사용자의 질문을 변경해주세요.
        만약 변경할 필요가 없다고 판단된다면, 사용자의 질문을 변경하지 않아도 됩니다.
        그런 경우에는 질문만 리턴해주세요.
        사전: {dictionary}
                                              
        질문: {{question}}
    """)

    chain = prompt | llm | StrOutputParser()
    new_question = chain.invoke({
        "question": query
    })

    tax_chain = {"query": chain} | qa_chain
    ai_response = tax_chain.invoke({"question": query})
    return ai_response['result']
