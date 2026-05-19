# src/agents/api.py  — HƯỚNG 1: Structured LLM Output
import json
import re
import calendar


class APIAgent:

    def __init__(self, llm_service, retriever, fewshot_loader=None):
        self.llm      = llm_service
        self.retriever = retriever
        self.fewshot  = fewshot_loader

    # DATE / ORG: rule-based 

    ORG_ALIASES = {
        "ttpmqt": "TTPMQT", "ttpmtcs": "TTPMTCS", "ttpmvt": "TTPMVT",
        "ttpmcnm": "TTPMCNM", "ttpmcds": "TTPMCDS", "ttcndt": "TTCNDT",
        "ttcnđt": "TTCNDT", "tt cnđt": "TTCNDT",
        "pm qt": "TTPMQT", "pm tcs": "TTPMTCS", "pm vt": "TTPMVT",
        "pm cnm": "TTPMCNM", "pm cds": "TTPMCDS",
        "ttpmcđs": "TTPMCDS", "tt pmcds": "TTPMCDS",
        "tt pmvt": "TTPMVT", "tt pmqt": "TTPMQT",
        "tt pmtcs": "TTPMTCS", "tt pmcnm": "TTPMCNM",
        "trung tâm pm qt": "TTPMQT", "trung tâm pm tcs": "TTPMTCS",
        "trung tâm pm vt": "TTPMVT", "trung tâm pm cnm": "TTPMCNM",
        "trung tâm pm cds": "TTPMCDS", "trung tâm cnđt": "TTCNDT",
    }

    def _ms(self, yr, mo):
        mo = max(1, min(12, mo))
        return f"{yr}-{mo:02d}-01"

    def _me(self, yr, mo):
        mo = max(1, min(12, mo))
        return f"{yr}-{mo:02d}-{calendar.monthrange(yr, mo)[1]}"

    def _extract_dates(self, question):
        q = question.lower()
        m = re.search(
            r"(?:tháng|t)\s*(\d{1,2})[/\-](\d{4})\s*(?:-|->|đến|~)\s*"
            r"(?:tháng|t)?\s*(\d{1,2})[/\-](\d{4})", q)
        if m:
            return self._ms(int(m.group(2)), int(m.group(1))), \
                   self._me(int(m.group(4)), int(m.group(3)))
        m = re.search(r"tháng\s*(\d{1,2})[/\-](20\d{2})", q)
        if m:
            return self._ms(int(m.group(2)), int(m.group(1))), \
                   self._me(int(m.group(2)), int(m.group(1)))
        m = re.search(r"\bt(\d{1,2})[/\-](20\d{2})\b", q)
        if m:
            return self._ms(int(m.group(2)), int(m.group(1))), \
                   self._me(int(m.group(2)), int(m.group(1)))
        m = re.search(r"(?:quý|q)\s*([1-4])[/\-\s](20\d{2})", q)
        if m:
            qtr, yr = int(m.group(1)), int(m.group(2))
            sm = (qtr - 1) * 3 + 1
            return self._ms(yr, sm), self._me(yr, sm + 2)
        m = re.search(r"(?:năm|year)\s*(20\d{2})", q)
        if m:
            yr = int(m.group(1))
            return f"{yr}-01-01", f"{yr}-12-31"
        m = re.search(r"\b(20\d{2})\b", q)
        if m:
            yr = int(m.group(1))
            return f"{yr}-01-01", f"{yr}-12-31"
        return "", ""

    def _extract_orgs(self, question):
        q = question.lower()
        found, seen = [], set()
        for alias, canon in self.ORG_ALIASES.items():
            if alias in q and canon not in seen:
                found.append(canon)
                seen.add(canon)
        return found

    # STRUCTURED PROMPT: LLM fill body theo schema 

    _SYSTEM = (
        "Bạn là AI assistant chuyên điền tham số cho API dashboard nội bộ. "
        "Chỉ trả về JSON hợp lệ, không giải thích, không markdown."
    )

    _PROMPT_TMPL = """\
[THÔNG TIN ĐÃ TRÍCH XUẤT]
fromDate: {from_date}
toDate: {to_date}
organization (danh sách đơn vị tìm thấy): {orgs}

[DANH SÁCH API ỨNG VIÊN]
{api_list}

[CÂU HỎI]
{question}

[NHIỆM VỤ]
1. Chọn func_code của API phù hợp nhất với câu hỏi.
2. Điền body params đúng theo spec của API đó.
   - Dùng fromDate/toDate đã cho ở trên.
   - Dùng organization đã cho ở trên (nếu API có param này).
   - Các param còn lại: đọc description trong spec và câu hỏi để điền.
   - Nếu param là list mà không đề cập → [].
   - Nếu param là nullable mà không đề cập → null.
   - KHÔNG thêm key nào không có trong spec.

Chỉ trả về JSON sau, không thêm bất cứ gì:
{{
  "func_code": "<func_code>",
  "path": "<path từ spec>",
  "body": {{ <chỉ các key có trong spec> }}
}}
"""

    def _build_api_list_str(self, top_df) -> str:
        parts = []
        for _, row in top_df.iterrows():
            try:
                cfg = json.loads(row["Endpoint config"])
            except Exception:
                continue
            req = cfg.get("request", {})
            all_params = cfg.get("required_params", []) + cfg.get("optional_params", [])
            # Lấy example body để LLM hiểu default values
            ex_calls = cfg.get("example_call", [])
            ex_body = ""
            if ex_calls:
                try:
                    ex_body = f"\n  example_body: {ex_calls[0].get('body', '')}"
                except Exception:
                    pass
            params_str = json.dumps(
                [{"name": p["name"], "type": p.get("type", ""), "description": p.get("description", "")}
                 for p in all_params],
                ensure_ascii=False
            )
            parts.append(
                f"func_code: {row['func_code']}\n"
                f"  path: {req.get('path', '')}\n"
                f"  method: {req.get('method', 'POST')}\n"
                f"  mô_tả: {str(row.get('description', ''))[:150]}\n"
                f"  params: {params_str}{ex_body}"
            )
        return "\n---\n".join(parts)

    def _call_llm_structured(self, question, from_date, to_date, orgs, top_df) -> dict:
        fewshot = ""
        if self.fewshot is not None:
            fewshot = self.fewshot.get_api_fewshot(question)

        api_list_str = self._build_api_list_str(top_df)
        prompt = (
            (f"{fewshot}\n" if fewshot else "")
            + self._PROMPT_TMPL.format(
                from_date=from_date or "không rõ",
                to_date=to_date or "không rõ",
                orgs=json.dumps(orgs, ensure_ascii=False),
                api_list=api_list_str,
                question=question,
            )
        )

        raw = self.llm.generate(prompt, max_tokens=400).strip()

        # Parse JSON từ output
        try:
            start = raw.find("{")
            end   = raw.rfind("}") + 1
            if start != -1 and end > start:
                return json.loads(raw[start:end])
        except Exception:
            pass

        # Fallback: thử tìm func_code ít nhất
        for _, row in top_df.iterrows():
            if row["func_code"].lower() in raw.lower():
                return {"func_code": row["func_code"], "path": "", "body": {}}

        return {}

    # POST-PROCESS: merge date/org đã extract chắc chắn vào LLM output

    def _merge_certain_fields(self, llm_result: dict, from_date, to_date, orgs, top_df) -> dict:
        """
        Override các field LLM hay sai bằng rule-based đã chắc chắn.
        LLM quyết định type/sort/path — rule quyết định date/org.
        """
        body = llm_result.get("body", {})

        # Date: rule luôn thắng
        for k in ("fromDate", "from_date", "startDate"):
            if k in body and from_date:
                body[k] = from_date
        for k in ("toDate", "to_date", "endDate"):
            if k in body and to_date:
                body[k] = to_date

        # Org: rule luôn thắng nếu param tồn tại trong body
        if "organization" in body:
            body["organization"] = orgs
        if "orgAlias" in body:
            body["orgAlias"] = orgs

        # isAllCustomer: normalize key case
        if "IsAllCustomer" in body:
            body["isAllCustomer"] = body.pop("IsAllCustomer")

        # projectType: normalize case
        TYPE_NORM = {
            "Package": "package", "PACKAGE": "package",
            "osdc": "odc/osdc", "OSDC": "odc/osdc", "ODC": "odc/osdc",
        }
        if "projectType" in body and isinstance(body["projectType"], list):
            body["projectType"] = [TYPE_NORM.get(v, v) for v in body["projectType"]]

        # Path: lấy từ LLM nếu có, nếu không lấy từ top API
        if not llm_result.get("path"):
            fc = llm_result.get("func_code", "")
            rows = top_df[top_df["func_code"] == fc]
            if not rows.empty:
                try:
                    cfg = json.loads(rows.iloc[0]["Endpoint config"])
                    llm_result["path"] = cfg.get("request", {}).get("path", "")
                except Exception:
                    pass

        llm_result["body"] = body
        return llm_result

    # MAIN 

    def process(self, question: str) -> str:
        top_df = self.retriever.get_top_apis_df(question, k=5)
        if top_df.empty:
            return "{}"

        from_date, to_date = self._extract_dates(question)
        orgs = self._extract_orgs(question)

        llm_result = self._call_llm_structured(question, from_date, to_date, orgs, top_df)
        if not llm_result:
            return "{}"

        llm_result = self._merge_certain_fields(llm_result, from_date, to_date, orgs, top_df)

        path = llm_result.get("path", "")
        body = llm_result.get("body", {})

        # GET API: body rỗng
        fc = llm_result.get("func_code", "")
        rows = top_df[top_df["func_code"] == fc]
        if not rows.empty:
            try:
                cfg = json.loads(rows.iloc[0]["Endpoint config"])
                if cfg.get("request", {}).get("method", "POST") == "GET":
                    return json.dumps({"path": path, "body": {}}, ensure_ascii=False)
            except Exception:
                pass

        return json.dumps({"path": path, "body": body}, ensure_ascii=False)