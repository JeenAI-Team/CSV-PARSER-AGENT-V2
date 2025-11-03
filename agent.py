"""
LangChain ReAct Agent for CSV/XLSX Analysis
Uses vLLM with Gemma to generate and execute Python code for data analysis.
Custom implementation compatible with vLLM middleware format.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List
import json
import traceback
import threading
import requests
import re


class CustomVLLM:
    """
    Custom LLM wrapper for vLLM with middleware format.
    Compatible with ChatOpenAI interface for vLLM endpoints.
    """
    
    def __init__(
        self,
        vllm_endpoint: str,
        model_deployment: str,
        temperature: float = 0.0,
        max_tokens: int = 2048
    ):
        self.vllm_endpoint = vllm_endpoint
        self.model_deployment = model_deployment
        self.temperature = temperature
        self.max_tokens = max_tokens
    
    def call(self, prompt: str, system_message: Optional[str] = None) -> str:
        """Call vLLM with custom middleware format"""
        
        # Clean model name (remove vllm- prefix)
        clean_model = self.model_deployment.replace("google/", "").replace("vllm-", "")
        
        # Default system message if not provided
        if not system_message:
            system_message = "You are a helpful AI assistant that follows instructions precisely."
        
        # Build payload in middleware format
        payload = {
            "model": clean_model,
            "inputs": [
                {
                    "role": "system",
                    "value": [
                        {
                            "type": "text",
                            "text": system_message
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
            "stream": False,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens
        }
        
        try:
            response = requests.post(
                self.vllm_endpoint,
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=60
            )
            
            if response.status_code == 200:
                data = response.json()
                # Handle middleware response format
                if "message" in data and "content" in data["message"]:
                    return data["message"]["content"]
                elif "choices" in data:
                    return data["choices"][0]["message"]["content"]
                else:
                    return str(data)
            else:
                raise Exception(f"vLLM request failed: {response.status_code} - {response.text[:200]}")
                
        except Exception as e:
            raise Exception(f"vLLM call failed: {str(e)}")


class DataAnalysisAgent:
    """
    LangChain-based agent that analyzes CSV/XLSX files using ReAct pattern.
    Generates clean Python code and retries automatically on errors.
    Thread-safe for parallel requests.
    """
    
    def __init__(
        self,
        vllm_endpoint: str,
        model_deployment: str,
        max_retries: int = 6,
        temperature: float = 0.1
    ):
        """
        Initialize the data analysis agent.
        
        Args:
            vllm_endpoint: vLLM server endpoint URL (chat format)
            model_deployment: Model deployment name (e.g., 'google/gemma-3-12b-it')
            max_retries: Maximum retry attempts for code execution
            temperature: Model temperature for code generation
        """
        self.max_retries = max_retries
        self.vllm_endpoint = vllm_endpoint
        self.model_deployment = model_deployment
        self.temperature = temperature
        
        # Thread-local storage for parallel request handling
        self._thread_local = threading.local()
        
        # Initialize LLM (using CustomVLLM for vLLM compatibility)
        self.llm = CustomVLLM(
            vllm_endpoint=vllm_endpoint,
            model_deployment=model_deployment,
            temperature=0.0,  # Force deterministic outputs for format compliance
            max_tokens=2048
        )
    
    def _get_thread_state(self):
        """Get thread-local state for parallel request handling"""
        if not hasattr(self._thread_local, 'state'):
            self._thread_local.state = {
                'current_df': None,
                'file_preview': None,
                'execution_history': []
            }
        return self._thread_local.state
    
    def _reset_thread_state(self):
        """Reset thread-local state"""
        self._thread_local.state = {
            'current_df': None,
            'file_preview': None,
            'execution_history': []
        }
    
    def _load_file(self, file_path: str) -> str:
        """Load CSV or Excel file and preview its structure"""
        state = self._get_thread_state()
        try:
            # Detect file type and load
            if file_path.lower().endswith(('.xlsx', '.xls')):
                state['current_df'] = pd.read_excel(file_path)
            else:
                state['current_df'] = pd.read_csv(file_path)
            
            # Create simple preview
            preview_df = state['current_df'].head(5)
            
            state['file_preview'] = {
                "total_rows": len(state['current_df']),
                "total_columns": len(state['current_df'].columns),
                "columns": list(state['current_df'].columns),
                "dtypes": {col: str(dtype) for col, dtype in state['current_df'].dtypes.items()},
                "sample_data": preview_df.to_dict(orient='records')
            }
            
            # Return structured format
            result = f"""File loaded successfully!
Rows: {state['file_preview']['total_rows']}
Columns: {', '.join(state['file_preview']['columns'])}
Sample (first 5 rows):
{json.dumps(state['file_preview']['sample_data'][:3], indent=2, ensure_ascii=False)}"""
            
            return result
        
        except Exception as e:
            return f"Error loading file: {str(e)}"
    
    def _execute_code(self, code: str) -> str:
        """Execute Python code to analyze the loaded DataFrame"""
        state = self._get_thread_state()
        
        if state['current_df'] is None:
            return "Error: No file loaded. Use 'load_file' first."
        
        # Clean code
        code = code.strip()
        if code.startswith("```python"):
            code = code.replace("```python", "").replace("```", "").strip()
        elif code.startswith("```"):
            code = code.replace("```", "").strip()
        
        # Validate: Reject code that tries to create plots
        forbidden_imports = ['matplotlib', 'plotly', 'seaborn', 'plt', 'pyplot']
        forbidden_functions = ['plt.', '.plot(', '.savefig(', '.figure(', '.show(']
        
        code_lower = code.lower()
        for forbidden in forbidden_imports:
            if f'import {forbidden}' in code_lower or f'from {forbidden}' in code_lower:
                return f"Error: Cannot import '{forbidden}'. Return tabular data instead - the frontend will create visualizations."
        
        for forbidden in forbidden_functions:
            if forbidden.lower() in code_lower:
                return f"Error: Cannot use '{forbidden}'. Return aggregated DataFrame using groupby().reset_index() - the frontend Canvas will visualize it."
        
        # Create namespace
        namespace = {
            "df": state['current_df'],
            "pd": pd,
            "np": np
        }
        
        try:
            # Execute
            exec(code, namespace)
            
            # Check for result
            if "result" not in namespace:
                error_msg = "Error: Code must assign final answer to 'result' variable"
                state['execution_history'].append({
                    "code": code,
                    "error": error_msg,
                    "success": False
                })
                return error_msg
            
            result = namespace["result"]
            formatted = self._format_result(result)
            
            # Record success
            state['execution_history'].append({
                "code": code,
                "result": formatted,
                "success": True
            })
            
            return f"✓ Code executed successfully!\nResult:\n{formatted}"
        
        except Exception as e:
            error_msg = f"Execution Error: {str(e)}"
            state['execution_history'].append({
                "code": code,
                "error": error_msg,
                "success": False
            })
            return error_msg
    
    def _format_result(self, result: Any) -> str:
        """Format result for display"""
        if isinstance(result, pd.DataFrame):
            # Return as JSON for Canvas/visualization tools
            try:
                return json.dumps(result.to_dict(orient='records'), ensure_ascii=False)
            except:
                # Fallback to string if JSON serialization fails
                return result.to_string()
        elif isinstance(result, pd.Series):
            try:
                return json.dumps(result.to_dict(), ensure_ascii=False)
            except:
                return result.to_string()
        elif isinstance(result, (np.integer, np.floating)):
            return str(result.item())
        elif isinstance(result, np.ndarray):
            return str(result.tolist())
        else:
            return str(result)
    
    def _parse_action(self, text: str) -> Optional[Dict[str, str]]:
        """Parse action from LLM output"""
        # Look for Action and Action Input
        action_match = re.search(r'Action:\s*(\w+)', text, re.IGNORECASE)
        input_match = re.search(r'Action Input:\s*(.+?)(?=\n\n|\nThought:|\nAction:|\nFinal Answer:|$)', text, re.DOTALL | re.IGNORECASE)
        
        if action_match:
            action = action_match.group(1).strip()
            action_input = input_match.group(1).strip() if input_match else ""
            return {"action": action, "input": action_input}
        
        return None
    
    def _parse_final_answer(self, text: str) -> Optional[str]:
        """Parse final answer from LLM output"""
        match = re.search(r'Final Answer:\s*(.+?)$', text, re.DOTALL | re.IGNORECASE)
        if match:
            return match.group(1).strip()
        return None
    
    def _extract_last_user_question(self, question: str) -> str:
        """
        Extract only the last user question from conversation history.
        Backend sends full conversation like:
        "User: old question\nAssistant: old answer\nUser: new question"
        We need only "new question"
        """
        # If question contains conversation history, extract last user message
        if 'User:' in question or 'Assistant:' in question:
            lines = question.split('\n')
            user_messages = []
            
            for line in lines:
                if line.strip().startswith('User:'):
                    # Extract message after "User:"
                    msg = line.split('User:', 1)[1].strip()
                    user_messages.append(msg)
            
            if user_messages:
                last_question = user_messages[-1]
                print(f"\n[INFO] Extracted last user question from conversation history")
                print(f"[INFO] Full conversation length: {len(question)} chars")
                print(f"[INFO] Extracted question: {last_question[:100]}...")
                return last_question
        
        # No conversation history, return as-is
        return question
    
    def analyze(self, file_path: str, question: str) -> Dict[str, Any]:
        """
        Analyze file and answer question using ReAct pattern.
        
        Args:
            file_path: Path to CSV or Excel file
            question: Question to answer (may include conversation history)
            
        Returns:
            Analysis results
        """
        # Reset state
        self._reset_thread_state()
        state = self._get_thread_state()
        
        # Extract only the last user question (ignore conversation history)
        clean_question = self._extract_last_user_question(question)
        
        # Build initial prompt
        system_message = """You are a data analysis assistant using ReAct pattern with REAL tool execution.

ABSOLUTE CRITICAL RULES - READ CAREFULLY:
1. You ONLY write these 3 lines, then STOP:
   - Thought: [your thinking]
   - Action: [tool name]
   - Action Input: [input]
   
2. After writing "Action Input:", you MUST STOP IMMEDIATELY
3. DO NOT write "Observation:" - the SYSTEM writes it
4. DO NOT continue after Action Input - WAIT
5. The system will add the real Observation from actual code execution
6. Only after seeing the system's Observation, write your next Thought

WHAT YOU WRITE (then STOP):
Thought: I need to load the file
Action: load_file
Action Input: /path/to/file.xlsx

[STOP HERE - DO NOT WRITE ANYTHING ELSE]

WHAT THE SYSTEM ADDS (not you):
Observation: File loaded successfully! Rows: 100, Columns: ...

THEN you continue:
Thought: Now I can analyze
Action: execute_python
Action Input: result = df['col'].sum()

[STOP AGAIN]

IF YOU WRITE "Observation:" YOURSELF, THE DATA WILL BE WRONG!"""
        
        prompt = f"""You are an expert data analyst that writes clean, concise Python code to analyze CSV/Excel files.

Available tools:
1. load_file - Load a CSV or Excel file and preview its structure
   Input: file_path (string)
   Output: JSON with columns, dtypes, and first rows sample
   Use this FIRST to understand the data structure

2. execute_python - Execute Python code to analyze the loaded DataFrame
   Input: Python code (string) that uses variable 'df' for the DataFrame
   Output: Execution result or error message
   The code MUST assign the final answer to variable 'result'
   Keep code concise and professional
   
   EXAMPLES OF GOOD CODE (with context):
   - result = df['column_name'].sum()  # Simple sum
   - result = df.groupby('category')['value'].mean()  # Returns category with mean
   - result = df.groupby('quarter')['sales'].sum()  # Returns quarter with sum
   - result = df.nlargest(5, 'amount')[['name', 'amount']]  # Top 5 with names
   - result = df.loc[df['value'].idxmax()]  # Row with max value (includes all columns)
   - result = df.groupby('month')['revenue'].agg(['sum', 'mean'])  # Multiple stats with month
   
   FOR CHARTS/GRAPHS/CANVAS - Return clean tabular data:
   - result = df.groupby('quarter')['sales'].sum().reset_index()  # For bar/line chart
   - result = df.groupby('category')['count'].sum().reset_index()  # For pie chart
   - result = df[['date', 'value']].sort_values('date')  # For time series
   - result = df.groupby(['quarter', 'category'])['sales'].sum().reset_index()  # For grouped chart
   
   CRITICAL: When finding max/min/average values:
   - DON'T return just the number: result = df['sales'].max()  ❌
   - DO return with context: result = df.loc[df['sales'].idxmax(), ['quarter', 'sales']]  ✓
   - DO use groupby to preserve labels: result = df.groupby('category')['value'].sum()  ✓
   
   EXAMPLES OF BAD CODE (DON'T DO THIS):
   - print(df['column'].mean())  # Don't print, assign to result
   - def calculate(): return df['col'].mean()  # Don't just define function
   - df['column'].mean()  # Don't leave hanging expression
   - result = df['sales'].max()  # Missing context - which row/quarter?
   - import matplotlib.pyplot as plt  # NEVER import plotting libraries!
   - plt.figure(); plt.plot(...); plt.savefig(...)  # NEVER create plots yourself!

IMPORTANT RULES:
1. ALWAYS start by using 'load_file' to understand the data structure
2. Write SHORT, CLEAN Python code - no verbose or poorly structured code
3. The DataFrame is available as 'df' variable
4. Your code MUST assign the final answer to variable 'result'
5. If code fails, analyze the error and fix it (don't repeat same mistakes)
6. Final Answer must be in Hebrew - clear, natural, and human-readable
7. Focus only on the user's question, no meta-explanations

CRITICAL - Context in Calculations:
8. When calculating max, min, average, sum - ALWAYS include context (row labels, category names, etc.)
9. Use df.loc[df['col'].idxmax()] to get the full row for max/min values
10. Use groupby() to preserve category/group labels with aggregated values
11. NEVER return just a number - return it WITH its source/context
12. Don't guess which quarter/row - extract it directly from the DataFrame columns

CRITICAL - For Charts/Graphs/Canvas:
13. NEVER import matplotlib, plotly, or any plotting library
14. NEVER call plt.figure(), plt.plot(), plt.savefig(), etc.
15. When user asks for chart/graph/canvas - return TABULAR DATA using groupby().reset_index()
16. The frontend Canvas will create the visualization - you only provide clean data
17. Your Final Answer for Canvas should be SHORT: "הנתונים מוכנים לגרף של X לפי Y"
18. Don't describe the data values in Final Answer - Canvas will display them

IMPORTANT: After you write "Action Input:", STOP and wait. The system will provide "Observation:".

Example conversation showing context retrieval:
Question: Which quarter had the highest sales?
Thought: I need to load the file first to see its structure
Action: load_file
Action Input: /path/to/file.xlsx

[YOU MUST STOP HERE - SYSTEM WILL RESPOND WITH OBSERVATION]

After seeing columns like 'quarter', 'sales':
Thought: I need to find max sales WITH the quarter name, not just the number
Action: execute_python
Action Input: result = df.loc[df['sales'].idxmax(), ['quarter', 'sales']]

[STOP - WAIT FOR OBSERVATION]

This returns BOTH the quarter name AND the sales value!

Another example for Canvas/Chart request:
Question: Create a pie chart of sales by category
Thought: User wants a chart, so I need to return aggregated tabular data
Action: execute_python
Action Input: result = df.groupby('category')['sales'].sum().reset_index()

[STOP - WAIT FOR OBSERVATION]

This returns a clean DataFrame with category and sales columns - perfect for Canvas to visualize!
NEVER use matplotlib or plt - just return the aggregated data!

Use this EXACT format:

Question: the input question you must answer
Thought: think about what to do
Action: the action to take (load_file or execute_python)  
Action Input: the input to the action

[STOP HERE - WAIT FOR SYSTEM'S OBSERVATION]

After receiving "Observation:" from the system, continue:
Thought: [analyze the observation]
Action: [next action]
Action Input: [input]

[STOP AGAIN - WAIT FOR NEXT OBSERVATION]

When you have the answer:
Thought: I now have all the information needed
Final Answer: [answer in Hebrew - clear and natural]

Now begin:

Question: {clean_question}
File: {file_path}

"""
        
        scratchpad = ""
        max_iterations = 15
        
        for iteration in range(max_iterations):
            state['iteration'] = iteration + 1
            
            # Call LLM
            full_prompt = prompt + scratchpad
            
            # Debug: Print what we're sending to LLM
            if iteration == 0:
                print(f"\n[DEBUG] Initial prompt length: {len(prompt)} chars")
                print(f"\n[DEBUG] First 500 chars of prompt:")
                print(full_prompt[:500])
                print("...")
            else:
                print(f"\n[DEBUG] Iteration {iteration + 1} - Scratchpad length: {len(scratchpad)} chars")
                print(f"\n[DEBUG] Last 300 chars of scratchpad:")
                print(scratchpad[-300:])
            
            try:
                response = self.llm.call(full_prompt, system_message)
            except Exception as e:
                return {
                    "status": "error",
                    "error": f"LLM call failed: {str(e)}",
                    "attempts": iteration + 1
                }
            
            print(f"\n[Iteration {iteration + 1}] LLM Response:")
            print(response)
            print("="*80)
            
            # Check for final answer
            final_answer = self._parse_final_answer(response)
            if final_answer:
                # Get successful execution
                successful_execution = None
                for entry in reversed(state['execution_history']):
                    if entry.get('success'):
                        successful_execution = entry
                        break
                
                # For successful execution, return both formatted answer and raw result
                result_data = successful_execution.get('result') if successful_execution else None
                
                return {
                    "status": "success",
                    "answer": final_answer,
                    "executed_code": successful_execution.get('code') if successful_execution else None,
                    "result": result_data,  # JSON formatted DataFrame for Canvas
                    "attempts": iteration + 1,
                    "file_info": state['file_preview']
                }
            
            # Parse action
            action_dict = self._parse_action(response)
            if not action_dict:
                scratchpad += f"\n{response}\n\nError: No valid Action found. Follow the format:\nThought: ...\nAction: ...\nAction Input: ...\n\n"
                continue
            
            action_name = action_dict["action"]
            action_input = action_dict["input"]
            
            # CRITICAL: Truncate response to remove any hallucinated Observations
            # Only keep response UP TO and INCLUDING the Action Input line
            response_lines = response.split('\n')
            truncated_response = []
            found_action_input = False
            
            for line in response_lines:
                # Stop if we see Observation (model hallucinating)
                if 'Observation:' in line or 'observation:' in line.lower():
                    print(f"[WARNING] Model hallucinated Observation - cutting it out!")
                    break
                
                truncated_response.append(line)
                
                # Mark that we found Action Input, take one more line max then stop
                if line.strip().startswith('Action Input:'):
                    found_action_input = True
                elif found_action_input:
                    # Already added Action Input line, now stop (allow max 1 blank line after)
                    if line.strip():  # If there's any content after Action Input, stop
                        break
            
            clean_response = '\n'.join(truncated_response).strip()
            
            if len(clean_response) < len(response):
                print(f"\n[DEBUG] Truncated hallucinated content: {len(response) - len(clean_response)} chars removed")
            print(f"[DEBUG] Clean response length: {len(clean_response)} chars")
            
            # Execute action
            observation = ""
            if action_name.lower() == "load_file":
                observation = self._load_file(action_input)
                print(f"\n[DEBUG] load_file returned: {observation[:200]}...")
            elif action_name.lower() == "execute_python":
                observation = self._execute_code(action_input)
                print(f"\n[DEBUG] execute_python returned: {observation[:200]}...")
            else:
                observation = f"Error: Unknown tool '{action_name}'. Use 'load_file' or 'execute_python'"
            
            print(f"\n[DEBUG] Adding to scratchpad - Observation length: {len(observation)}")
            
            # Add to scratchpad - use clean_response without hallucinated observations
            scratchpad += f"\n{clean_response}\nObservation: {observation}\n\n"
            
            print(f"\n[DEBUG] Scratchpad now has {len(scratchpad)} chars")
        
        # Max iterations reached
        return {
            "status": "error",
            "error": "Maximum iterations reached without finding answer",
            "attempts": max_iterations,
            "execution_history": state['execution_history']
        }


# Convenience function
def analyze_file(
    file_path: str,
    question: str,
    vllm_endpoint: str,
    model_deployment: str,
    **kwargs
) -> Dict[str, Any]:
    """Quick function to analyze a file"""
    agent = DataAnalysisAgent(
        vllm_endpoint=vllm_endpoint,
        model_deployment=model_deployment,
        **kwargs
    )
    return agent.analyze(file_path, question)
