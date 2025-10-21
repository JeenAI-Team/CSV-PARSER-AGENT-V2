"""
Clean CSV/XLSX Analysis Agent
Simple microservice that generates and executes Python code
Compatible with vLLM endpoints
"""

import pandas as pd
import numpy as np
import json
import traceback
import requests
from typing import Dict, Any, Optional


class CSVAnalysisAgent:
    """
    Simple agent for CSV/XLSX analysis.
    Generates Python code and executes it on data.
    """
    
    def __init__(
        self,
        vllm_endpoint: str,
        model_deployment: str,
        max_retries: int = 3,
        temperature: float = 0.1
    ):
        """
        Initialize agent.
        
        Args:
            vllm_endpoint: vLLM /chat endpoint URL
            model_deployment: Model name (e.g., 'google/gemma-3-12b-it')
            max_retries: Max retry attempts
            temperature: Model temperature
        """
        self.vllm_endpoint = vllm_endpoint
        self.model_deployment = model_deployment
        self.max_retries = max_retries
        self.temperature = temperature
        
        print(f"\n[CSV Agent] Initialized")
        print(f"- Endpoint: {vllm_endpoint}")
        print(f"- Model: {model_deployment}\n")
    
    def analyze(self, file_path: str, question: str) -> Dict[str, Any]:
        """
        Analyze CSV or Excel file.
        
        Args:
            file_path: Path to file
            question: Question to answer
            
        Returns:
            Analysis results
        """
        print(f"\n{'='*80}")
        print(f"[ANALYSIS] {question}")
        print(f"{'='*80}\n")
        
        try:
            # Load file
            df = self._load_file(file_path)
            print(f"✓ Loaded {df.shape[0]} rows × {df.shape[1]} columns")
            
            # Get preview (first 10 rows)
            preview = self._get_preview(df)
            
            # Generate code with retries
            code = None
            for attempt in range(1, self.max_retries + 1):
                print(f"\nAttempt {attempt}/{self.max_retries}...")
                code = self._generate_code(preview, question)
                
                if code and self._validate_code(code, df):
                    print(f"✓ Code validated")
                    break
                else:
                    print(f"✗ Code invalid, retrying...")
                    code = None
            
            if not code:
                return {
                    "status": "error",
                    "error": "Failed to generate valid code",
                    "attempts": self.max_retries
                }
            
            # Execute code
            print(f"\n{code}\n")
            result_value = self._execute_code(code, df)
            print(f"✓ Result: {str(result_value)[:100]}...")
            
            # Generate explanation
            explanation = self._explain(question, result_value)
            
            print(f"✓ Complete!\n")
            
            return {
                "status": "success",
                "answer": explanation,
                "executed_code": code,
                "result": self._format_result(result_value),
                "attempts": attempt,
                "file_info": preview,
                "explanation": explanation
            }
            
        except Exception as e:
            print(f"\n[ERROR] {str(e)}\n")
            traceback.print_exc()
            
            return {
                "status": "error",
                "error": str(e),
                "traceback": traceback.format_exc()
            }
    
    def _load_file(self, file_path: str) -> pd.DataFrame:
        """Load CSV or Excel file"""
        if file_path.lower().endswith(('.xlsx', '.xls')):
            return pd.read_excel(file_path)
        return pd.read_csv(file_path)
    
    def _get_preview(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Get first 10 rows preview"""
        preview_df = df.head(10)
        
        return {
            "total_rows": len(df),
            "total_columns": len(df.columns),
            "columns": list(df.columns),
            "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
            "sample_data": preview_df.to_dict(orient='records'),
            "shape": df.shape
        }
    
    def _generate_code(self, preview: Dict[str, Any], question: str) -> Optional[str]:
        """Generate Python code using vLLM"""
        
        # VERY STRONG prompt that prevents DataFrame creation
        # and converts chart intent into tabular aggregation output
        prompt = f"""CRITICAL: DataFrame is ALREADY LOADED as 'df'. DO NOT create a new DataFrame!

DATA ALREADY IN 'df':
Columns: {preview['columns']}
Total rows: {preview['total_rows']}
Sample (first 2 rows): {json.dumps(preview['sample_data'][:2], ensure_ascii=False)}

QUESTION: {question}

STRICT RULES:
1. DataFrame 'df' is ALREADY loaded with ALL {preview['total_rows']} rows
2. DO NOT create new DataFrame with pd.DataFrame()
3. DO NOT use pd.read_csv() or pd.read_excel()
4. ONLY use the existing 'df' variable
5. Write 1-2 concise lines
6. Assign the final answer to 'result'

IF THE USER ASKS FOR A CHART/GRAPH (e.g., contains 'chart', 'graph', or 'גרף'):
- DO NOT plot and DO NOT import visualization libraries
- INSTEAD compute a compact aggregation suitable for plotting and assign it to 'result'
- Prefer a two-column DataFrame: one categorical label column and one numeric value column
- Examples:
  result = df.groupby('airline').size().reset_index(name='count')
  result = df.groupby('destination_city')['price'].mean().reset_index(name='avg_price')

GENERAL EXAMPLES (using existing df):
result = df['price'].sum()
result = df.groupby('category')['value'].mean()
result = df[df['age'] > 18].shape[0]

WRONG (DON'T DO THIS):
df = pd.DataFrame(...)  ← NO! df already exists
result = pd.read_csv(...)  ← NO! Data already loaded

YOUR CODE (use existing df only):"""
        
        response = self._call_vllm(prompt)
        
        if not response:
            return None
        
        return self._extract_code(response)
    
    def _call_vllm(self, prompt: str) -> Optional[str]:
        """Call vLLM endpoint using custom middleware format"""
        
        # Use custom format that your vLLM middleware expects
        clean_model = self.model_deployment.replace("google/", "").replace("vllm-", "")
        
        # Simple system prompt that works with the middleware
        system_prompt = """You are a Python code generator for data analysis.
Your ONLY job is to generate executable Python code.
Never generate JSON, HTML, or use tools.
Always respond with pure Python code."""
        
        payload = {
            "model": clean_model,
            "inputs": [
                {
                    "role": "system",
                    "value": [
                        {
                            "type": "text",
                            "text": system_prompt
                        }
                    ]
                },
                {
                    "role": "user",
                    "value": [
                        {
                            "type": "text",
                            "text": prompt
                        }
                    ]
                }
            ],
            "stream": False
        }
        
        try:
            response = requests.post(
                self.vllm_endpoint,
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=30
            )
            
            if response.status_code == 200:
                data = response.json()
                # Handle middleware response format
                if "message" in data and "content" in data["message"]:
                    return data["message"]["content"]
                elif "choices" in data:
                    return data["choices"][0]["message"]["content"]
            
            print(f"  vLLM error: {response.status_code} - {response.text[:200]}")
            return None
            
        except Exception as e:
            print(f"  Request failed: {str(e)}")
            return None
    
    def _extract_code(self, response: str) -> Optional[str]:
        """Extract Python code from response"""
        
        # If response is JSON (Canvas format), reject it
        if response.strip().startswith('{') or 'tool_call' in response or '"name"' in response:
            print("  ✗ Response is JSON/Canvas format, rejecting")
            return None
        
        # Remove markdown
        response = response.replace('```python', '```').replace('```json', '')
        
        # Extract from code blocks
        if '```' in response:
            parts = response.split('```')
            for part in parts:
                # Skip JSON parts
                if 'tool_call' in part or '"name"' in part:
                    continue
                if 'result' in part and '=' in part:
                    lines = [l.strip() for l in part.split('\n') if l.strip()]
                    code_lines = [l for l in lines if not l.startswith('#') or 'result' in l]
                    if code_lines:
                        return '\n'.join(code_lines)
        
        # Find result assignment in raw text
        for line in response.split('\n'):
            line = line.strip()
            if line.startswith('result') and '=' in line and not line.startswith('#'):
                # Make sure it's not inside JSON
                if '{' not in line and '"' not in line:
                    return line
        
        return None
    
    def _validate_code(self, code: str, df: pd.DataFrame) -> bool:
        """Test if code works and doesn't create new DataFrames"""
        
        # REJECT code that creates new DataFrames (sign of hallucination)
        forbidden_patterns = [
            'pd.DataFrame(',
            'DataFrame(',
            'pd.read_csv(',
            'pd.read_excel(',
        ]
        
        for pattern in forbidden_patterns:
            if pattern in code:
                print(f"  ✗ Code creates new DataFrame - REJECTED")
                return False
        
        # MUST use the existing 'df' variable
        if 'df[' not in code and 'df.' not in code and 'df ' not in code:
            print(f"  ✗ Code doesn't use existing 'df' - REJECTED")
            return False
        
        try:
            namespace = {"df": df, "pd": pd, "np": np}
            exec(code, namespace)
            return "result" in namespace
        except:
            return False
    
    def _execute_code(self, code: str, df: pd.DataFrame) -> Any:
        """Execute code and return result"""
        namespace = {"df": df, "pd": pd, "np": np}
        exec(code, namespace)
        return namespace["result"]
    
    def _explain(self, question: str, result: Any) -> str:
        """Generate simple explanation in Hebrew"""
        result_str = str(result)[:200]
        # Keep it concise and always in Hebrew
        return f"עניתי לשאלה: '{question}'. התוצאה היא: {result_str}"
    
    def _format_result(self, result: Any) -> str:
        """Format result for JSON"""
        if isinstance(result, pd.DataFrame):
            return result.to_string()
        elif isinstance(result, pd.Series):
            return result.to_string()
        elif isinstance(result, (np.integer, np.floating)):
            return str(result.item())
        elif isinstance(result, np.ndarray):
            return str(result.tolist())
        return str(result)

