import argparse
import json
import re
import os
import pandas as pd


def _read(path: str) -> pd.DataFrame:
    if path.endswith((".xlsx", ".xls")):
        return pd.read_excel(path)
    for enc in ["utf-8", "utf-8-sig", "cp1252", "latin-1"]:
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception:
            continue
    raise ValueError(f"Không đọc được: {path}")


def _load_example(example_path: str):
    xls    = pd.ExcelFile(example_path)
    sheets = xls.sheet_names
    q_sheet = next((s for s in sheets if "question" in s.lower()), sheets[0])
    a_sheet = next((s for s in sheets if "result"   in s.lower()
                                      or "answer"   in s.lower()), None)
    df_q = pd.read_excel(example_path, sheet_name=q_sheet)
    df_a = pd.read_excel(example_path, sheet_name=a_sheet) if a_sheet else None
    return df_q, df_a


def _norm_letters(s: str) -> set:
    return set(re.findall(r"[ABCD]", str(s).upper()))


def _parse_json_safe(s: str) -> dict:
    try:
        return json.loads(s)
    except Exception:
        return {}


def score_func_code(pred: str, gold: str) -> int:
    return int(str(pred).strip().lower() == str(gold).strip().lower())


def score_doc_answer(pred_param: str, gold_param: str) -> float:
    pred_letters = _norm_letters(_parse_json_safe(pred_param).get("result", pred_param))
    gold_letters = _norm_letters(_parse_json_safe(gold_param).get("result", gold_param))
    if not gold_letters:
        return 0.0
    if pred_letters == gold_letters:
        return 1.0
    if pred_letters & gold_letters:
        return 0.5
    return 0.0


def score_api_answer(pred_param: str, gold_param: str) -> float:
    pred = _parse_json_safe(pred_param)
    gold = _parse_json_safe(gold_param)
    if not pred or not gold:
        return 0.0

    # Path (50%)
    pred_path = str(pred.get("path", "")).strip().rstrip("/")
    gold_path = str(gold.get("path", "")).strip().rstrip("/")
    path_score = 0.5 if pred_path == gold_path else 0.0

    # Body fields (50%)
    pred_body = pred.get("body", {}) or {}
    gold_body = gold.get("body", {}) or {}
    if isinstance(pred_body, str): pred_body = _parse_json_safe(pred_body)
    if isinstance(gold_body, str): gold_body = _parse_json_safe(gold_body)

    KEY_FIELDS = [
        "fromDate", "toDate", "organization", "orgAlias",
        "projectType", "projectStatus", "type", "sort",
        "standardComparison", "isProbation", "targetCode", "cycleType",
    ]
    gold_keys = [k for k in KEY_FIELDS if k in gold_body]
    body_score = 0.0
    if gold_keys:
        matched = sum(
            1 for k in gold_keys
            if str(pred_body.get(k, "")).strip() == str(gold_body.get(k, "")).strip()
        )
        body_score = 0.5 * (matched / len(gold_keys))

    return round(path_score + body_score, 4)


def evaluate(result_path: str, example_path: str):
    df_result = _read(result_path)
    df_result["id"] = df_result["id"].astype(str)

    _, df_gold = _load_example(example_path)
    if df_gold is None:
        print("❌ Không tìm thấy sheet kết quả trong example file.")
        return

    df_gold["id"] = df_gold["id"].astype(str)
    df = df_gold.merge(df_result, on="id", suffixes=("_gold", "_pred"))
    if df.empty:
        print("❌ Không có id nào khớp giữa result và example.")
        return

    print(f"\n📊 Đánh giá {len(df)}/{len(df_gold)} câu example\n")

    gold_code  = next(c for c in df.columns if "func_code"  in c and "gold" in c)
    pred_code  = next(c for c in df.columns if "func_code"  in c and "pred" in c)
    gold_param = next(c for c in df.columns if "func_param" in c and "gold" in c)
    pred_param = next(c for c in df.columns if "func_param" in c and "pred" in c)

    rows = []
    for _, row in df.iterrows():
        gc = str(row[gold_code]).strip()
        pc = str(row[pred_code]).strip()
        gp = str(row.get(gold_param, ""))
        pp = str(row.get(pred_param, ""))

        code_ok = score_func_code(pc, gc)
        if not code_ok:
            param_score = 0.0
        elif gc == "call_document":
            param_score = score_doc_answer(pp, gp)
        else:
            param_score = score_api_answer(pp, gp)

        rows.append({
            "id":          row["id"],
            "gold_code":   gc,
            "pred_code":   pc,
            "code_ok":     code_ok,
            "param_score": param_score,
            "total":       round(0.4 * code_ok + 0.6 * param_score, 4),
        })

    df_scores = pd.DataFrame(rows)
    doc = df_scores[df_scores["gold_code"] == "call_document"]
    api = df_scores[df_scores["gold_code"] == "call_api"]

    print("=" * 50)
    print(f"  Tổng câu              : {len(df_scores)}")
    print(f"  Func_code accuracy    : {df_scores['code_ok'].mean()*100:.1f}%")
    print(f"  Param score TB        : {df_scores['param_score'].mean()*100:.1f}%")
    print(f"  Điểm tổng (weighted)  : {df_scores['total'].mean()*100:.1f}%")
    print()
    if not doc.empty:
        print(f"  [call_document] {len(doc)} câu")
        print(f"    func_code acc : {doc['code_ok'].mean()*100:.1f}%")
        print(f"    param score   : {doc['param_score'].mean()*100:.1f}%")
        print(f"    tổng          : {doc['total'].mean()*100:.1f}%")
    if not api.empty:
        print(f"  [call_api]      {len(api)} câu")
        print(f"    func_code acc : {api['code_ok'].mean()*100:.1f}%")
        print(f"    param score   : {api['param_score'].mean()*100:.1f}%")
        print(f"    tổng          : {api['total'].mean()*100:.1f}%")
    print("=" * 50)

    # Kaggle: output về /kaggle/working
    out_dir  = os.path.dirname(result_path) or "/kaggle/working"
    out_path = os.path.join(out_dir, "eval_detail.csv")
    df_scores.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"\n💾 Chi tiết lưu tại: {out_path}")

    wrong = df_scores[df_scores["total"] < 0.5].sort_values("total")
    if not wrong.empty:
        print(f"\n⚠️  {len(wrong)} câu điểm < 0.5:")
        print(wrong[["id", "gold_code", "pred_code", "total"]].head(10).to_string(index=False))

    return df_scores


if __name__ == "__main__":
    # Đường dẫn mặc định theo cấu trúc Kaggle
    KAGGLE_RESULT  = "/kaggle/working/result.csv"
    KAGGLE_EXAMPLE = "/kaggle/input/ai-race-data/example_data/example_data.xlsx"

    parser = argparse.ArgumentParser()
    parser.add_argument("--result",  default=KAGGLE_RESULT)
    parser.add_argument("--example", default=KAGGLE_EXAMPLE)
    args = parser.parse_args()

    evaluate(args.result, args.example)