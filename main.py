import json
import os
import re
import csv
import shutil
import time
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer

from src.llm import LLMService
from src.router import RouterAgent
from src.fewshot import FewShotLoader
from src.agents.document import DocAgent
from src.agents.api import APIAgent
from src.retrieval.apiretriever import APIRetriever

# CONFIG
INPUT_FILE       = "./data/test_data/Test_data.xlsx"
EXAMPLE_FILE     = "./data/example_data/example_data.xlsx"
OUTPUT_FILE      = "./data/output/result.csv"
CHECKPOINT_FILE  = "./data/output/checkpoint.csv"
EVAL_OUTPUT_FILE = "./data/output/eval_results.csv"
INDEX_DIR        = "./data/knowledge"
API_CSV          = "./data/API_config_data/api_config.csv"
EXAMPLE_DIR      = "./data/example_data"
USE_ENSEMBLE     = False

TOP_K_API      = 5
TOP_K_RETRIEVE = 10
TOP_K_RERANK   = 7


def _read_input(path: str) -> pd.DataFrame:
    if path.endswith((".xlsx", ".xls")):
        return pd.read_excel(path)
    for enc in ["utf-8", "utf-8-sig", "cp1252", "latin-1"]:
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception:
            continue
    raise ValueError(f"Không đọc được file: {path}")


def load_services():
    print("=" * 60)
    print("1/5 Khởi tạo LLM...")
    llm = LLMService()
    test_out = llm.generate("1+1 bằng mấy? Trả lời:", max_tokens=10)
    print(f" Test: '{test_out}'")
    if not test_out.strip():
        raise RuntimeError("❌ LLM load thất bại!")

    print("2/5 Embedding model...")
    embed_model = SentenceTransformer("keepitreal/vietnamese-sbert")

    print("3/5 FewShotLoader...")
    fewshot = FewShotLoader(example_dir=EXAMPLE_DIR, embed_model=embed_model, top_k=2)

    print("4/5 Router + DocAgent + APIAgent...")
    router        = RouterAgent(llm_service=llm)
    doc_agent     = DocAgent(llm_service=llm, index_dir=INDEX_DIR,
                             fewshot_loader=fewshot, use_ensemble=USE_ENSEMBLE)
    api_retriever = APIRetriever(api_csv_path=API_CSV, embed_model=embed_model)
    api_agent     = APIAgent(llm_service=llm, retriever=api_retriever, fewshot_loader=fewshot)

    print("5/5 Sẵn sàng.")
    print("=" * 60)
    return router, doc_agent, api_agent


def _parse_note(note_raw) -> str:
    if note_raw is None:
        return ""
    if isinstance(note_raw, float) and pd.isna(note_raw):
        return ""
    s = str(note_raw).strip()
    return "" if s.lower() == "nan" else s


def process_row(row, router, doc_agent, api_agent) -> dict:
    qid      = str(row.get("id", ""))
    question = str(row.get("fun_question", row.get("question", ""))).strip()

    t0 = time.time()

    func_code = router.classify(question)

    if func_code == "call_document":
        note_str   = _parse_note(row.get("note", ""))
        raw_result = doc_agent.process(question, note=note_str)
    else:
        raw_result = api_agent.process(question)

    elapsed = round(time.time() - t0, 3)

    try:
        parsed = json.loads(raw_result)
        func_param = json.dumps(parsed, ensure_ascii=False)
    except Exception:
        func_param = str(raw_result)

    return {
        "id":         qid,
        "func_code":  func_code,
        "func_param": func_param,
        "time":       elapsed,
    }


def load_checkpoint() -> set:
    if os.path.exists(CHECKPOINT_FILE):
        try:
            df   = _read_input(CHECKPOINT_FILE)
            done = set(df["id"].astype(str).tolist())
            print(f"🔄 Checkpoint: đã xử lý {len(done)} câu.")
            return done
        except Exception:
            pass
    return set()


def save_checkpoint(results: list):
    os.makedirs(os.path.dirname(CHECKPOINT_FILE), exist_ok=True)
    pd.DataFrame(results).to_csv(CHECKPOINT_FILE, index=False, encoding="utf-8-sig")
    print(f"💾 Checkpoint: {len(results)} câu.")


def is_match_params(truth, pred):
    """Subset Matching cho JSON: pred chứa đủ key/value của truth là True."""
    if isinstance(truth, dict) and isinstance(pred, dict):
        for key, expected_value in truth.items():
            if key not in pred:
                return False
            if not is_match_params(expected_value, pred[key]):
                return False
        return True
    elif isinstance(truth, list) and isinstance(pred, list):
        if len(truth) != len(pred):
            return False
        for t_item, p_item in zip(truth, pred):
            if not is_match_params(t_item, p_item):
                return False
        return True
    else:
        return str(truth).strip() == str(pred).strip()


def eval(router, doc_agent, api_agent):
    print("Bắt đầu evaluation trên tập dữ liệu Train...")

    xls    = pd.ExcelFile(EXAMPLE_FILE)
    sheets = xls.sheet_names
    q_sheet = next((s for s in sheets if "question" in s.lower()), sheets[0])
    a_sheet = next((s for s in sheets if "result"   in s.lower()
                                      or "answer"   in s.lower()), None)
    if a_sheet is None:
        print("Lỗi: Không tìm thấy sheet kết quả trong example file.")
        return

    df_train    = pd.read_excel(EXAMPLE_FILE, sheet_name=q_sheet)
    df_train_rs = pd.read_excel(EXAMPLE_FILE, sheet_name=a_sheet)

    if 'note' not in df_train.columns:
        df_train['note'] = ""

    df_merged = pd.merge(df_train, df_train_rs, on='id', how='inner')

    results         = []
    correct_count   = 0
    total_questions = len(df_merged)

    print(f"Bắt đầu xử lý và đánh giá {total_questions} câu hỏi...\n")

    for index, row in df_merged.iterrows():
        start_time  = time.time()
        question    = str(row.get("fun_question", row.get("question", ""))).strip()
        q_id        = str(row['id'])
        truth_param = str(row['func_param']).strip()

        # Bước 1: router chỉ nhận question
        func_code = router.classify(question)

        # Bước 2: chỉ lấy note sau khi đã route sang call_document
        if func_code == "call_document":
            note_str   = _parse_note(row.get("note", ""))
            pred_param = doc_agent.process(question, note=note_str)
        else:
            pred_param = api_agent.process(question)

        try:
            pred_param = json.dumps(json.loads(pred_param), ensure_ascii=False)
        except Exception:
            pred_param = str(pred_param)

        time_response = int((time.time() - start_time) * 1000)

        is_correct = False
        try:
            pred_dict  = json.loads(pred_param)
            truth_dict = json.loads(truth_param)
            is_correct = is_match_params(truth_dict, pred_dict)
        except Exception:
            is_correct = (truth_param in pred_param)

        if is_correct:
            correct_count += 1

        results.append({
            "id":             q_id,
            "question":       question,
            "func_code":      func_code,
            "predicted_param": pred_param,
            "truth_param":    truth_param,
            "is_correct":     is_correct,
            "time_response":  time_response
        })

        if (index + 1) % 10 == 0:
            current_acc = (correct_count / (index + 1)) * 100
            print(f"Đã xử lý {index + 1}/{total_questions} câu... Acc tạm thời: {current_acc:.2f}%")

    final_accuracy = (correct_count / total_questions) * 100 if total_questions > 0 else 0

    print("=" * 50)
    print("BÁO CÁO KẾT QUẢ ĐÁNH GIÁ (EVALUATION)")
    print(f"Tổng số câu hỏi    : {total_questions}")
    print(f"Số câu trả lời đúng: {correct_count}")
    print(f"Độ chính xác       : {final_accuracy:.2f}%")
    print("=" * 50)

    os.makedirs(os.path.dirname(EVAL_OUTPUT_FILE), exist_ok=True)
    pd.DataFrame(results).to_csv(
        EVAL_OUTPUT_FILE,
        index=False,
        encoding='utf-8-sig',
        quoting=csv.QUOTE_ALL
    )
    print(f"Chi tiết đúng/sai từng câu đã được lưu tại: {EVAL_OUTPUT_FILE}")

    return final_accuracy


def main(router=None, doc_agent=None, api_agent=None):
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    df_input = _read_input(INPUT_FILE)
    print(f"📋 Tổng: {len(df_input)} câu")

    if router is None or doc_agent is None or api_agent is None:
        router, doc_agent, api_agent = load_services()

    done_ids = load_checkpoint()
    results: list = []
    if done_ids:
        results = _read_input(CHECKPOINT_FILE).to_dict("records")

    remaining = df_input[~df_input["id"].astype(str).isin(done_ids)]
    print(f"▶️  Còn {len(remaining)} câu.\n")

    for i, (_, row) in enumerate(remaining.iterrows(), 1):
        try:
            result = process_row(row, router, doc_agent, api_agent)
            results.append(result)
            print(f"[{i}/{len(remaining)}] id={result['id']} → {result['func_code']} ({result['time']}s)")
        except Exception as e:
            print(f"⚠️ Lỗi id={row.get('id', '?')}: {e}")
            results.append({
                "id":         str(row.get("id", "")),
                "func_code":  "call_document",
                "func_param": '{"numbers": 1, "result": "A"}',
                "time":       0.0,
            })
        if i % 30 == 0:
            save_checkpoint(results)

    df_out = pd.DataFrame(results)
    df_out.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")
    save_checkpoint(results)
    print(f"\n✅ Xong! {OUTPUT_FILE}")
    print(df_out.head(5).to_string(index=False))


if __name__ == "__main__":
    router, doc_agent, api_agent = load_services()

    print("Chọn options:\n"
          "0 - eval + infer\n"
          "1 - Chỉ eval\n"
          "2 - Chỉ infer")
    n = input("Nhập lựa chọn của bạn: ").strip()

    if n == "0":
        eval(router, doc_agent, api_agent)
        main(router, doc_agent, api_agent)
    elif n == "1":
        eval(router, doc_agent, api_agent)
    elif n == "2":
        main(router, doc_agent, api_agent)
    else:
        print("Lựa chọn không hợp lệ. Vui lòng chạy lại và chọn 0, 1 hoặc 2.")