import re

class RouterAgent:
    def classify(self, question):
        text = str(question).lower()
        
        strict_api_keywords = [
            r'\bttpm\w*\b', r'\bcbnv\b', r'\bnslđ\b', r'leakage rate', 
            r'defect rate', r'\bosdc\b', r'\bpackage\b', r'tr đồng', 
            r'trđ', r'mm/người', r'\bslnt\b', r'\bslsx\b', r'\bcpnc\b', r'\blcnt\b'
        ]
        for pattern in strict_api_keywords:
            if re.search(pattern, text): return "call_api"
                
        time_patterns = [
            r'trong năm 202\d', r'trong t\d{1,2}/202\d', 
            r'trong tháng \d{1,2}\s*/202\d', r'trong quý \d/202\d', r'q\d/202\d',
            r'(?:tháng|t)\s*\d{1,2}/\d{4}\s*(?:-|->|đến)\s*(?:tháng|t)\s*\d{1,2}/\d{4}'
        ]
        for pattern in time_patterns:
            if re.search(pattern, text): return "call_api"
                
        return "call_document"