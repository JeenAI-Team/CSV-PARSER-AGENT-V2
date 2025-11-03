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
            
            # Generate code with retries and error feedback
            code = None
            last_error = None
            
            for attempt in range(1, self.max_retries + 1):
                print(f"\nAttempt {attempt}/{self.max_retries}...")
                
                # Pass previous error to help agent self-correct
                code = self._generate_code(preview, question, previous_error=last_error)
                
                # Try to execute and validate
                validation_result = self._validate_code_with_execution(code, df)
                
                if validation_result['valid']:
                    print(f"✓ Code validated and executed successfully")
                    # Return the successful result immediately
                    result = validation_result['result']
                    explanation = self._explain_result(question, code, result)
                    
                    print(f"✓ Result: {str(result)[:200]}...")
                    print(f"✓ Complete!\n")
                    
                    return {
                        "status": "success",
                        "result": self._format_result(result),
                        "code": code,
                        "file_info": preview,
                        "explanation": explanation,
                        "attempts": attempt
                    }
                else:
                    print(f"✗ Code execution failed: {validation_result['error']}")
                    last_error = validation_result['error']
                    code = None
            
            # If all attempts failed, return the last error
            return {
                "status": "error",
                "error": f"Failed after {self.max_retries} attempts. Last error: {last_error}",
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
        """Load CSV or Excel file with encoding detection"""
        if file_path.lower().endswith(('.xlsx', '.xls')):
            return pd.read_excel(file_path)
        
        # Try multiple encodings for CSV files (common for Hebrew files)
        encodings_to_try = ['utf-8', 'windows-1255', 'iso-8859-8', 'cp1252', 'latin1']
        
        for encoding in encodings_to_try:
            try:
                df = pd.read_csv(file_path, encoding=encoding)
                print(f"✓ Successfully loaded CSV with encoding: {encoding}")
                return df
            except (UnicodeDecodeError, UnicodeError):
                continue
            except Exception as e:
                # If it's not an encoding error, raise it
                if "codec can't decode" not in str(e):
                    raise
        
        # If all encodings fail, try with errors='ignore'
        print("⚠️ Warning: Using fallback encoding with errors='ignore'")
        return pd.read_csv(file_path, encoding='utf-8', errors='ignore')
    
    def _get_preview(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Get safe DataFrame preview for JSON serialization"""
        # Get sample data and convert NaN/NaT to None for JSON serialization
        sample_data = df.head(10).replace({pd.NaT: None, np.nan: None}).to_dict(orient='records')
        
        # Clean sample data - ensure all values are JSON serializable
        cleaned_sample = []
        for row in sample_data:
            cleaned_row = {}
            for key, value in row.items():
                if pd.isna(value) or value is pd.NaT:
                    cleaned_row[key] = None
                elif isinstance(value, (np.integer, np.floating)):
                    cleaned_row[key] = float(value) if np.isnan(value) else value.item()
                elif isinstance(value, (pd.Timestamp, pd.Timedelta)):
                    cleaned_row[key] = str(value)
                else:
                    cleaned_row[key] = value
            cleaned_sample.append(cleaned_row)
        
        return {
            "columns": list(df.columns),
            "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
            "sample_data": cleaned_sample,
            "total_rows": len(df),
            "shape": df.shape
        }
    
    def _generate_code(self, preview: Dict[str, Any], question: str, previous_error: Optional[str] = None) -> Optional[str]:
        """Generate Python code using vLLM with optional error feedback for self-correction"""
        
        # Build error feedback section
        error_feedback = ""
        if previous_error:
            error_feedback = f"""
PREVIOUS ATTEMPT FAILED WITH ERROR:
{previous_error}

Please fix the error and try again. Common issues:
- KeyError: Check column names match exactly
- ValueError: Ensure data types are compatible
- TypeError: Handle NaN/None values properly
"""
        
        # VERY STRONG prompt that prevents DataFrame creation
        # and converts chart intent into tabular aggregation output
        prompt = f"""CRITICAL: DataFrame is ALREADY LOADED as 'df'. DO NOT create a new DataFrame!

DATA ALREADY IN 'df':
Columns: {preview['columns']}
Total rows: {preview['total_rows']}
Sample (first 2 rows): {json.dumps(preview['sample_data'][:2], ensure_ascii=False)}
{error_feedback}
QUESTION: {question}

STRICT RULES:
1. DataFrame 'df' is ALREADY loaded with ALL {preview['total_rows']} rows
2. DO NOT create new DataFrame with pd.DataFrame()
3. DO NOT use pd.read_csv() or pd.read_excel()
4. ONLY use the existing 'df' variable
5. Write 1-2 concise lines
6. Assign the final answer to 'result'
7. Answer in Hebrew only
8. Handle NaN/None values gracefully with .fillna() or .dropna() if needed

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
    
    def _validate_code_with_execution(self, code: str, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Test if code works, execute it, and return result with error feedback.
        This combines validation and execution for better self-correction.
        """
        if not code:
            return {"valid": False, "error": "No code generated", "result": None}
        
        # REJECT code that creates new DataFrames (sign of hallucination)
        forbidden_patterns = [
            'pd.DataFrame(',
            'DataFrame(',
            'pd.read_csv(',
            'pd.read_excel(',
        ]
        
        for pattern in forbidden_patterns:
            if pattern in code:
                return {
                    "valid": False,
                    "error": f"Code creates new DataFrame using '{pattern}' - must use existing 'df' variable only",
                    "result": None
                }
        
        # MUST use the existing 'df' variable
        if 'df[' not in code and 'df.' not in code and 'df ' not in code:
            return {
                "valid": False,
                "error": "Code doesn't reference existing 'df' variable - must use 'df' to access the loaded data",
                "result": None
            }
        
        # Try to execute the code
        try:
            namespace = {"df": df, "pd": pd, "np": np}
            exec(code, namespace)
            
            if "result" not in namespace:
                return {
                    "valid": False,
                    "error": "Code executed but didn't assign to 'result' variable",
                    "result": None
                }
            
            return {
                "valid": True,
                "error": None,
                "result": namespace["result"]
            }
            
        except KeyError as e:
            return {
                "valid": False,
                "error": f"KeyError: Column {str(e)} not found. Available columns: {list(df.columns)}",
                "result": None
            }
        except ValueError as e:
            return {
                "valid": False,
                "error": f"ValueError: {str(e)}. Check data types and handle NaN values",
                "result": None
            }
        except Exception as e:
            return {
                "valid": False,
                "error": f"{type(e).__name__}: {str(e)}",
                "result": None
            }
    
    def _validate_code(self, code: str, df: pd.DataFrame) -> bool:
        """DEPRECATED: Use _validate_code_with_execution instead"""
        result = self._validate_code_with_execution(code, df)
        return result['valid']
    
    def _execute_code(self, code: str, df: pd.DataFrame) -> Any:
        """Execute code and return result"""
        namespace = {"df": df, "pd": pd, "np": np}
        exec(code, namespace)
        return namespace["result"]
    
    def _explain(self, question: str, result: Any) -> str:
        """Generate simple explanation in Hebrew"""
        result_str = str(result)[:200]
        # Return only the result without robotic phrasing
        return result_str
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

