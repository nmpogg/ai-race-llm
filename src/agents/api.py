import json
import re
import calendar


class APIAgent:

    def __init__(self, llm_service, retriever, fewshot_loader=None):
        self.llm       = llm_service
        self.retriever = retriever
        self.fewshot   = fewshot_loader

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

    # PROMPT: few-shot in-context 

    _PROMPT_TMPL = """\
{fewshot}

[THÔNG TIN ĐÃ BIẾT CHẮC]
fromDate: {from_date}
toDate:   {to_date}
organization: {orgs}

[DANH SÁCH API ỨNG VIÊN - chọn 1 cái phù hợp nhất]
{api_candidates}

[CÂU HỎI CẦN XỬ LÝ]
{question}

Học theo pattern từ các ví dụ trên. Điền đúng path và body params theo spec.
Dùng fromDate/toDate/organization đã cho ở trên.
Chỉ trả về JSON, không giải thích:
{{
  "path": "<path của API>",
  "body": {{ <params> }}
}}
JSON:"""

    def _build_candidates_str(self, top_df) -> str:
        parts = []
        for _, row in top_df.iterrows():
            try:
                cfg = json.loads(row["Endpoint config"])
            except Exception:
                continue
            req        = cfg.get("request", {})
            all_params = cfg.get("required_params", []) + cfg.get("optional_params", [])
            param_names = [
                f"{p['name']} ({p.get('description','')[:60]})"
                for p in all_params
            ]
            parts.append(
                f"• func_code: {row['func_code']}\n"
                f"  path: {req.get('path','')}\n"
                f"  mô tả: {str(row.get('description',''))[:120]}\n"
                f"  params: {', '.join(param_names)}"
            )
        return "\n".join(parts)

    # MAIN 

    def process(self, question: str) -> str:
        top_df = self.retriever.get_top_apis_df(question, k=5)
        if top_df.empty:
            return "{}"

        from_date, to_date = self._extract_dates(question)
        orgs = self._extract_orgs(question)

        # Few-shot: lấy 4 examples gần nhất (nhiều hơn để LLM học đủ pattern)
        fewshot_str = ""
        if self.fewshot is not None:
            fewshot_str = self.fewshot.get_api_fewshot(question, k=4)

        candidates_str = self._build_candidates_str(top_df)

        prompt = self._PROMPT_TMPL.format(
            fewshot=fewshot_str,
            from_date=from_date or "không rõ",
            to_date=to_date or "không rõ",
            orgs=json.dumps(orgs, ensure_ascii=False),
            api_candidates=candidates_str,
            question=question,
        )

        raw = self.llm.generate(prompt, max_tokens=400).strip()

        # Parse JSON
        result = {}
        try:
            start = raw.find("{")
            end   = raw.rfind("}") + 1
            if start != -1 and end > start:
                result = json.loads(raw[start:end])
        except Exception:
            pass

        if not result:
            # Fallback: dùng top API
            try:
                cfg  = json.loads(top_df.iloc[0]["Endpoint config"])
                path = cfg.get("request", {}).get("path", "")
                return json.dumps({"path": path, "body": {}}, ensure_ascii=False)
            except Exception:
                return "{}"

        # Normalize chắc chắn
        body = result.get("body", {})
        for k in ("fromDate", "from_date", "startDate"):
            if k in body and from_date:
                body[k] = from_date
        for k in ("toDate", "to_date", "endDate"):
            if k in body and to_date:
                body[k] = to_date
        if "organization" in body:
            body["organization"] = orgs
        if "orgAlias" in body:
            body["orgAlias"] = orgs
        if "IsAllCustomer" in body:
            body["isAllCustomer"] = body.pop("IsAllCustomer")

        TYPE_NORM = {"Package": "package", "PACKAGE": "package",
                     "osdc": "odc/osdc", "OSDC": "odc/osdc"}
        if "projectType" in body and isinstance(body["projectType"], list):
            body["projectType"] = [TYPE_NORM.get(v, v) for v in body["projectType"]]

        path = result.get("path", "")
        # Nếu LLM không trả path, lấy từ func_code
        if not path:
            fc_hint = result.get("func_code", "")
            rows = top_df[top_df["func_code"] == fc_hint]
            if not rows.empty:
                try:
                    cfg  = json.loads(rows.iloc[0]["Endpoint config"])
                    path = cfg.get("request", {}).get("path", "")
                except Exception:
                    pass

        return json.dumps({"path": path, "body": body}, ensure_ascii=False)