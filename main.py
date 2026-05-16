import os
import gc
import time
import csv
import pandas as pd
import json
import pickle
from src.preprocess.pdf2md import process_pdf_data
from src.preprocess.chunking import MarkdownChunker
from src.preprocess.build_index import build_index
from src.llm import LLMService
from src.router import RouterAgent
from src.retrieval.apiretriever import APIRetriever
from src.agents.api import APIAgent
from src.agents.document import DocAgent

DIR_PDF_INPUT     = "./data/Document_config_data"
DIR_MD_OUTPUT     = "./data/markdown"
FILE_MASTER_MD    = "./data/markdown/corpus.md"
DIR_CHUNK_DATA_JSON   = "./data/knowledge/chunks.json"
DIR_CHUNK_DATA_PKL   = "./data/knowledge/chunks.pkl"
DIR_INDEX_DATA    = "./data/knowledge" # faiss.index, chunks.pkl, bm25.pkl

RUN_PDF_TO_MD = not os.path.exists(FILE_MASTER_MD)
RUN_CHUNKING = not os.path.exists(DIR_CHUNK_DATA_JSON) and not os.path.exists(DIR_CHUNK_DATA_PKL)
RUN_BUILD_INDEX = not (
    os.path.exists(DIR_INDEX_DATA) and 
    os.path.exists(os.path.join(DIR_INDEX_DATA, "faiss.index"))
)

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
TOP_K_API = 5
TOP_K_RETRIEVE = 10
TOP_K_RERANK = 7

FILE_TRAIN_DATA   = "./data/example_data/example_data.xlsx"
FILE_TEST_DATA    = "./data/test_data/Test_data.xlsx"
FILE_API_CONFIG   = "./data/API_config_data/Tài liệu config API.xlsx"
FILE_EVAL_OUTPUT = "eval_results.csv"
FILE_SUBMISSION   = "submission.csv"

llm_service = None
router = None
retriever = None
api_agent = None
doc_agent = None
chunker = None

def load_service():
    global llm_service, router, retriever, api_agent, doc_agent
    
    print("🚀 KHỞI ĐỘNG HỆ THỐNG AI RACE PIPELINE...")
    
    if RUN_PDF_TO_MD:
        process_pdf_data(DIR_PDF_INPUT, DIR_MD_OUTPUT, FILE_MASTER_MD)

    chunks = None

    if RUN_CHUNKING:
        chunker = MarkdownChunker(max_chunk_size=3000)
        chunks = list(chunker.stream_and_chunk(FILE_MASTER_MD))
        print(f"Đã tạo thành công {len(chunks)} chunks!")

        # lưu toàn bộ list chunks ra file .pkl
        with open(DIR_CHUNK_DATA_PKL, "wb") as f:
            pickle.dump(chunks, f)
        
        # .json for check
        with open(DIR_CHUNK_DATA_JSON, "w", encoding="utf-8") as f:
            json.dump(chunks, f, ensure_ascii=False, indent=2)
            
        print(f"Đã lưu toàn bộ dữ liệu ra file: {DIR_CHUNK_DATA_PKL} và {DIR_CHUNK_DATA_JSON}")

    if RUN_BUILD_INDEX:
        if chunks is None:
            print("Đang nạp chunks từ file pickle để build index...")
            with open(DIR_CHUNK_DATA_PKL, "rb") as f:
                chunks = pickle.load(f)
        build_index(chunks, DIR_INDEX_DATA)

    del chunks
    gc.collect()
    
    # load service
    print("Loading LLM Service, Router, API Retriever, API Agent, Document Agent...")
    llm_service = LLMService(model_path=MODEL_NAME)
    router = RouterAgent()
    retriever = APIRetriever(FILE_API_CONFIG)
    
    api_agent = APIAgent(llm_service, retriever)
    doc_agent = DocAgent(llm_service, DIR_INDEX_DATA)
    print("Đã nạp thành công toàn bộ Services và Agents!")


def is_match_params(truth, pred):
    """
    Thuật toán kiểm tra Subset Matching cho JSON.
    Chỉ cần 'pred' (dự đoán) chứa đầy đủ các key/value của 'truth' (đáp án) là True.
    """
    # Xử lý nếu là Dictionary
    if isinstance(truth, dict) and isinstance(pred, dict):
        for key, expected_value in truth.items():
            if key not in pred:
                return False # Thiếu key so với đáp án gốc
            
            # Đệ quy kiểm tra value bên trong
            if not is_match_params(expected_value, pred[key]):
                return False
        return True
        
    # Xử lý nếu là Mảng (List)
    elif isinstance(truth, list) and isinstance(pred, list):
        if len(truth) != len(pred):
            return False
        # Kiểm tra từng phần tử trong mảng
        for t_item, p_item in zip(truth, pred):
            if not is_match_params(t_item, p_item):
                return False
        return True
        
    # Xử lý giá trị đơn (Chuỗi, Số nguyên, Float, Boolean)
    else:
        # Ép về chuỗi và cắt khoảng trắng 2 đầu để so sánh chống lỗi typo
        return str(truth).strip() == str(pred).strip()

def eval():
    global router, api_agent, doc_agent
    print("Bắt đầu evaluation trên tập dữ liệu Train...")
    
    if FILE_TRAIN_DATA.endswith('.xlsx'):
        df_train = pd.read_excel(FILE_TRAIN_DATA, sheet_name="example_question")
        df_train_rs = pd.read_excel(FILE_TRAIN_DATA, sheet_name="example_result")
    else:
        print("Lỗi: File Train phải là định dạng Excel (.xlsx) chứa 2 sheets.")
        return

    if 'note' not in df_train.columns: 
        df_train['note'] = ""

    df_merged = pd.merge(df_train, df_train_rs, on='id', how='inner')
    
    results = []
    correct_count = 0
    total_questions = len(df_merged)
    
    print(f"Bắt đầu xử lý và đánh giá {total_questions} câu hỏi...\n")
    
    for index, row in df_merged.iterrows():
        start_time = time.time()
        
        question = row['fun_question']
        note = row['note']
        q_id = row['id']
        truth_param = str(row['func_param']).strip()
        
        func_code = router.classify(question)
        
        if func_code == "call_api":
            pred_param = api_agent.process(question, top_k=TOP_K_API)
        else:
            pred_param = doc_agent.process(question, note, top_k_retrieve=TOP_K_RETRIEVE, top_k_rerank=TOP_K_RERANK)
            
        time_response = int((time.time() - start_time) * 1000)

        is_correct = False

        # parse to dictionary
        pred_dict = json.loads(str(pred_param))
        truth_dict = json.loads(truth_param)
        
        is_correct = is_match_params(truth_dict, pred_dict)     

        if is_correct:
            correct_count += 1

        results.append({
            "id": q_id,
            "question": question,
            "func_code": func_code,
            "predicted_param": pred_param,
            "truth_param": truth_param,
            "is_correct": is_correct,
            "time_response": time_response
        })
        
        if (index + 1) % 10 == 0: 
            current_acc = (correct_count / (index + 1)) * 100
            print(f"Đã xử lý {index + 1}/{total_questions} câu... Acc tạm thời: {current_acc:.2f}%")

    final_accuracy = (correct_count / total_questions) * 100 if total_questions > 0 else 0
    
    print("BÁO CÁO KẾT QUẢ ĐÁNH GIÁ (EVALUATION)")

    print(f"Tổng số câu hỏi : {total_questions}")
    print(f"Số câu trả lời đúng : {correct_count}")
    print(f"Độ chính xác (Accuracy) : {final_accuracy:.2f}%")

    pd.DataFrame(results).to_csv(
        FILE_EVAL_OUTPUT, 
        index=False, 
        encoding='utf-8-sig', 
        quoting=csv.QUOTE_ALL
    )
    print(f"Chi tiết đúng/sai từng câu đã được lưu tại: {FILE_EVAL_OUTPUT}")
    
    return final_accuracy

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
            "time_response": int((time.time() - start_time) * 1000)
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
    print("Chọn options:\n" \
    "0 - eval + infer\n" \
    "1 - Chỉ eval\n" \
    "2 - Chỉ infer")
    n = input("Nhập lựa chọn của bạn: ").strip()
    if n == "0":
        eval()
        infer()
    elif n == "1": 
        eval()
    elif n == "2":
        infer()
    else:        print("Lựa chọn không hợp lệ. Vui lòng chạy lại và chọn 0, 1 hoặc 2.")