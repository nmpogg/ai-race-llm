import json
import os
import time
import pandas as pd
from sentence_transformers import SentenceTransformer

from src.llm import LLMService
from src.router import RouterAgent
from src.fewshot import FewShotLoader
from src.agents.document import DocAgent
from src.agents.api import APIAgent
from src.retrieval.apiretriever import APIRetriever

# CONFIG 
INPUT_FILE       = "./data/test_data/Test_data.xlsx"
OUTPUT_FILE      = "./data/output/result.csv"
CHECKPOINT_FILE  = "./data/output/checkpoint.csv"
INDEX_DIR        = "./data/knowledge"
API_CSV          = "./data/API_config_data/api_config.csv"
EXAMPLE_DIR      = "./data/example_data"
USE_ENSEMBLE     = True

# ĐỌC FILE (hỗ trợ xlsx + csv nhiều encoding)
def _read_input(path: str) -> pd.DataFrame:
    if path.endswith((".xlsx", ".xls")):
        return pd.read_excel(path)
    for enc in ["utf-8", "utf-8-sig", "cp1252", "latin-1"]:
        try:
            return pd.read_csv(path, encoding=enc)
        except (UnicodeDecodeError, Exception):
            continue
    raise ValueError(f"Không đọc được file: {path}")

# LOAD SERVICES 
def load_services():
    print("=" * 60)
    print("1/5 Khởi tạo LLM (Qwen2.5-7B 4-bit)...")
    llm = LLMService()

    print("2/5 Khởi tạo Embedding model (dùng chung)...")
    embed_model = SentenceTransformer("keepitreal/vietnamese-sbert")

    print("3/5 Khởi tạo FewShotLoader từ example_data...")
    fewshot = FewShotLoader(
        example_dir=EXAMPLE_DIR,
        embed_model=embed_model,
        top_k=2,
    )

    print("4/5 Khởi tạo Router + DocAgent + APIAgent...")
    router = RouterAgent(llm_service=llm)

    doc_agent = DocAgent(
        llm_service=llm,
        index_dir=INDEX_DIR,
        fewshot_loader=fewshot,
        use_ensemble=USE_ENSEMBLE,
    )

    api_retriever = APIRetriever(
        api_csv_path=API_CSV,
        embed_model=embed_model,
    )

    api_agent = APIAgent(
        llm_service=llm,
        retriever=api_retriever,
        fewshot_loader=fewshot,
    )

    print("5/5 Tất cả services đã sẵn sàng.")
    print("=" * 60)
    return router, doc_agent, api_agent

# PROCESS ONE ROW 
def process_row(row, router, doc_agent, api_agent) -> dict:
    qid      = str(row.get("id", ""))
    question = str(row.get("question", "")).strip()
    note     = str(row.get("note", "")).strip()

    t0 = time.time()

    func_code = router.classify(question)

    if func_code == "call_document":
        raw_result = doc_agent.process(question, note=note)
    else:
        raw_result = api_agent.process(question)

    elapsed = round(time.time() - t0, 3)

    try:
        parsed = json.loads(raw_result)
        function_result = json.dumps(parsed, ensure_ascii=False)
    except Exception:
        function_result = str(raw_result)

    return {
        "id":              qid,
        "function_code":   func_code,
        "function_result": function_result,
        "time_response":   elapsed,
    }

# CHECKPOINT 
def load_checkpoint() -> set[str]:
    if os.path.exists(CHECKPOINT_FILE):
        try:
            df = _read_input(CHECKPOINT_FILE)
            done = set(df["id"].astype(str).tolist())
            print(f"🔄 Checkpoint: đã xử lý {len(done)} câu, tiếp tục...")
            return done
        except Exception:
            pass
    return set()

def save_checkpoint(results: list):
    os.makedirs(os.path.dirname(CHECKPOINT_FILE), exist_ok=True)
    pd.DataFrame(results).to_csv(CHECKPOINT_FILE, index=False, encoding="utf-8-sig")

# MAIN 
def main(router=None, doc_agent=None, api_agent=None):
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

    df_input = _read_input(INPUT_FILE)
    print(f"📋 Tổng số câu hỏi: {len(df_input)}")

    # Dùng services truyền vào, không load lại
    if router is None or doc_agent is None or api_agent is None:
        router, doc_agent, api_agent = load_services()

    done_ids = load_checkpoint()
    results: list = []
    if done_ids:
        results = _read_input(CHECKPOINT_FILE).to_dict("records")

    remaining = df_input[~df_input["id"].astype(str).isin(done_ids)]
    print(f"▶️  Còn {len(remaining)} câu cần xử lý.\n")

    for i, (_, row) in enumerate(remaining.iterrows(), 1):
        try:
            result = process_row(row, router, doc_agent, api_agent)
            results.append(result)
            print(
                f"[{i}/{len(remaining)}] id={result['id']} "
                f"→ {result['function_code']} "
                f"({result['time_response']}s)"
            )
        except Exception as e:
            print(f"⚠️ Lỗi câu id={row.get('id', '?')}: {e}")
            results.append({
                "id":              str(row.get("id", "")),
                "function_code":   "call_document",
                "function_result": '{"numbers": 1, "result": "A"}',
                "time_response":   0.0,
            })

        if i % 30 == 0:
            save_checkpoint(results)

    df_out = pd.DataFrame(results)
    df_out.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")
    save_checkpoint(results)
    print(f"\n✅ Hoàn thành! Output: {OUTPUT_FILE}")
    print(df_out.head(5).to_string(index=False))


if __name__ == "__main__":
    main()