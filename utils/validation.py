# utils/validation.py
import re
import html
from typing import Any

def validate_input(text: str) -> str:
    """Validate and sanitize user input"""
    if not isinstance(text, str):
        raise ValueError("Input must be a string")
    
    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text.strip())
    
    # HTML escape to prevent XSS
    text = html.escape(text)
    
    # Check for suspicious patterns
    suspicious_patterns = [
        r'<script.*?>.*?</script>',
        r'javascript:',
        r'on\w+\s*=',
        r'eval\s*\(',
        r'document\.',
        r'window\.'
    ]
    
    for pattern in suspicious_patterns:
        if re.search(pattern, text, re.IGNORECASE):
            raise ValueError("Input contains suspicious content")
    
    return text

def validate_sql_query(query: str) -> bool:
    """Validate SQL query for safety"""
    dangerous_keywords = [
        'DROP', 'DELETE', 'INSERT', 'UPDATE', 'CREATE', 'ALTER',
        'EXEC', 'EXECUTE', 'TRUNCATE', 'GRANT', 'REVOKE'
    ]
    
    upper_query = query.upper()
    for keyword in dangerous_keywords:
        if keyword in upper_query:
            return False
    
    return True
