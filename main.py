import os
import time
import csv
import pandas as pd
from src.preprocess.build_index import build_index
from src.preprocess.pdf2md import process_pdf_data
from src.llm import LLMService
from src.router import RouterAgent
from src.retrieval.apiretriever import APIRetriever
from src.agents.api import APIAgent
from src.agents.document import DocAgent

DIR_PDF_INPUT     = "./data/Document_config_data"
DIR_MD_OUTPUT     = "./data/markdown"
FILE_MASTER_MD    = "./data/markdown/corpus.md"
DIR_INDEX_DATA    = "./data/knowledge" # faiss.index, chunks.pkl, bm25.pkl

RUN_PDF_TO_MD = not os.path.exists(FILE_MASTER_MD)
RUN_BUILD_INDEX = not (
    os.path.exists(DIR_INDEX_DATA) and 
    os.path.exists(os.path.join(DIR_INDEX_DATA, "faiss.index"))
)

FILE_TRAIN_DATA   = "Train_data.xlsx - question_train.csv"
FILE_TEST_DATA    = "Test_data.xlsx - question_test.csv"
FILE_API_CONFIG   = "Tài liệu config API.xlsx - Doc_api_for_contest.csv"
FILE_SUBMISSION   = "submission.csv"

llm_service = None
router = None
retriever = None
api_agent = None
doc_agent = None

def load_service():
    global llm_service, router, retriever, api_agent, doc_agent
    
    print("🚀 KHỞI ĐỘNG HỆ THỐNG AI RACE PIPELINE...")
    
    if RUN_PDF_TO_MD:
        process_pdf_data(DIR_PDF_INPUT, DIR_MD_OUTPUT, FILE_MASTER_MD)
        
    if RUN_BUILD_INDEX:
        build_index(FILE_MASTER_MD, DIR_INDEX_DATA)
    
    # load service
    print("Loading LLM Service, Router, API Retriever, API Agent, Document Agent...")
    llm_service = LLMService()
    router = RouterAgent()
    retriever = APIRetriever(FILE_API_CONFIG)
    
    api_agent = APIAgent(llm_service, retriever)
    doc_agent = DocAgent(llm_service, DIR_INDEX_DATA)
    print("Đã nạp thành công toàn bộ Services và Agents!")

def eval():
    global router, api_agent, doc_agent
    print("Bắt đầu Evaluation trên tập dữ liệu Train...")
    # Logic eval 
    return

def infer():
    global router, api_agent, doc_agent 
    
    print("Bắt đầu Inference trên tập dữ liệu Test...")
    # read test data
    if FILE_TEST_DATA.endswith('.xlsx'):
        df_test = pd.read_excel(FILE_TEST_DATA)
    else:
        df_test = pd.read_csv(FILE_TEST_DATA)
        
    if 'note' not in df_test.columns: 
        df_test['note'] = ""
    
    results = []
    print(f"Bắt đầu xử lý {len(df_test)} câu hỏi...")
    
    for index, row in df_test.iterrows():
        start_time = time.time()
        
        question = row['fun_question']
        note = row['note']
        q_id = row['id']
        
        # router
        func_code = router.classify(question)
        
        if func_code == "call_api":
            func_param = api_agent.process(question)
        else:
            func_param = doc_agent.process(question, note)

        results.append({
            "id": q_id,
            "func_code": func_code,
            "func_param": func_param,
            "time": int((time.time() - start_time) * 1000)
        })
        
        if (index + 1) % 10 == 0: 
            print(f"Đã xử lý {index + 1}/{len(df_test)} câu...")

    # save submission file
    pd.DataFrame(results).to_csv(
        FILE_SUBMISSION, 
        index=False, 
        encoding='utf-8-sig', 
        quoting=csv.QUOTE_ALL
    )
    print(f"Success! File nộp bài đã lưu tại: {FILE_SUBMISSION}")
    
if __name__ == "__main__":
    load_service() 
    
    # eval()
    # infer()