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
INPUT_FILE       = "./test_data/input.csv"       # hoặc đường dẫn thực tế
OUTPUT_FILE      = "./output/result.csv"
CHECKPOINT_FILE  = "./output/checkpoint.csv"     # lưu tiến độ, chạy tiếp khi bị ngắt
INDEX_DIR        = "./index_data"
API_CSV          = "./API_config_data/api_config.csv"
EXAMPLE_DIR      = "./example_data"
USE_ENSEMBLE     = True                          # True = chạy 2 lần/câu, ổn định hơn

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
    router  = RouterAgent(llm_service=llm)

    doc_agent = DocAgent(
        llm_service=llm,
        index_dir=INDEX_DIR,
        fewshot_loader=fewshot,
        use_ensemble=USE_ENSEMBLE,
    )
    # Truyền embed_model dùng chung → không load lại model, tiết kiệm VRAM
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

    # Bước 1: Router phân loại câu hỏi
    func_code = router.classify(question)

    # Bước 2: Xử lý theo loại bài toán
    if func_code == "call_document":
        raw_result = doc_agent.process(question, note=note)
    else:
        raw_result = api_agent.process(question)

    elapsed = round(time.time() - t0, 3)

    # Bước 3: Parse function_result
    try:
        parsed = json.loads(raw_result)
        function_result = parsed
    except Exception:
        function_result = raw_result

    return {
        "id":              qid,
        "function_code":   func_code,
        "function_result": json.dumps(function_result, ensure_ascii=False)
                           if isinstance(function_result, dict)
                           else str(function_result),
        "time_response":   elapsed,
    }

# CHECKPOINT HELPERS 
def load_checkpoint() -> set[str]:
    """Trả về tập hợp các id đã xử lý (để bỏ qua khi chạy lại)."""
    if os.path.exists(CHECKPOINT_FILE):
        try:
            df = pd.read_csv(CHECKPOINT_FILE)
            done = set(df["id"].astype(str).tolist())
            print(f"🔄 Checkpoint: đã xử lý {len(done)} câu, tiếp tục từ câu còn lại...")
            return done
        except Exception:
            pass
    return set()

def save_checkpoint(results: list[dict]):
    """Lưu kết quả hiện tại vào file checkpoint."""
    os.makedirs(os.path.dirname(CHECKPOINT_FILE), exist_ok=True)
    pd.DataFrame(results).to_csv(CHECKPOINT_FILE, index=False, encoding="utf-8-sig")

# MAIN
def main():
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

    # Load input
    df_input = pd.read_csv(INPUT_FILE)
    print(f"📋 Tổng số câu hỏi: {len(df_input)}")

    # Load services
    router, doc_agent, api_agent = load_services()

    # Load checkpoint nếu có
    done_ids = load_checkpoint()
    results: list[dict] = []
    if done_ids:
        # Load lại kết quả đã có
        results = pd.read_csv(CHECKPOINT_FILE).to_dict("records")

    # Lọc các câu chưa xử lý
    remaining = df_input[~df_input["id"].astype(str).isin(done_ids)]
    print(f"▶️  Còn {len(remaining)} câu cần xử lý.\n")

    # Xử lý từng câu
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
                "func_code":   "call_document",
                "func_result": '{"numbers": 1, "result": "A"}',
                "time":   0.0,
            })

        # Lưu checkpoint mỗi 30 câu
        if i % 30 == 0:
            save_checkpoint(results)
            print(f"💾 Checkpoint đã lưu ({len(results)} câu).")

    # Lưu output cuối cùng
    df_out = pd.DataFrame(results)
    df_out.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")
    save_checkpoint(results)
    print(f"\n✅ Hoàn thành! Output: {OUTPUT_FILE}")
    print(df_out.head(5).to_string(index=False))


if __name__ == "__main__":
    main()