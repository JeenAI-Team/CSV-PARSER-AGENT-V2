"""
LangChain ReAct Agent for CSV/XLSX Analysis
Uses vLLM with Gemma to generate and execute Python code for data analysis.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List
from langchain.agents import create_react_agent, AgentExecutor
from langchain.prompts import PromptTemplate
from langchain.tools import Tool
from langchain_openai import ChatOpenAI
import json
import traceback
import threading


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
        max_retries: int = 3,
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
        
        # Initialize LLM (using ChatOpenAI for vLLM compatibility)
        self.llm = ChatOpenAI(
            openai_api_key="EMPTY",
            openai_api_base=vllm_endpoint,
            model_name=model_deployment,
            temperature=temperature,
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
    
    def _create_executor(self):
        """Create a new executor for each request (thread-safe)"""
        tools = self._create_tools()
        agent = self._create_agent(tools)
        return AgentExecutor(
            agent=agent,
            tools=tools,
            verbose=True,
            handle_parsing_errors=True,
            max_iterations=self.max_retries * 2
        )
    
    def _create_tools(self) -> List[Tool]:
        """Create tools for the agent to use."""
        
        return [
            Tool(
                name="load_file",
                func=self._load_file,
                description="""Load a CSV or Excel file and preview its structure.
                Input: file_path (string)
                Output: JSON with columns, dtypes, and first 10 rows sample.
                Use this FIRST to understand the data structure."""
            ),
            Tool(
                name="execute_python",
                func=self._execute_code,
                description="""Execute Python code to analyze the loaded DataFrame.
                Input: Python code (string) that uses variable 'df' for the DataFrame.
                Output: Execution result or error message.
                The code MUST assign the final answer to variable 'result'.
                Keep code concise and professional. Example:
                result = df['column'].mean()
                """
            ),
            Tool(
                name="get_execution_history",
                func=self._get_history,
                description="""Get history of previous code execution attempts.
                Input: None (pass empty string)
                Output: List of previous attempts with errors.
                Use this to avoid repeating the same mistakes."""
            )
        ]
    
    def _create_agent(self, tools: List[Tool]):
        """Create the ReAct agent with custom prompt."""
        
        template = """You are an expert data analyst that writes clean, concise Python code to analyze CSV/Excel files.

You have access to these tools:
{tools}

Tool Names: {tool_names}

IMPORTANT RULES:
1. ALWAYS start by using 'load_file' to understand the data structure
2. Write SHORT, CLEAN Python code - no verbose or poorly structured code
3. The DataFrame is available as 'df' variable
4. Your code MUST assign the final answer to variable 'result'
5. If code fails, analyze the error and fix it (don't repeat same mistakes)
6. Check execution history to avoid repeating errors
7.All outputs (Thought and Final Answer) must be in Hebrew only
8. When generating answers based on a document, always provide a clear, natural, and human-readable response
9.The response must focus only on the user’s question and the relevant content from the document, without meta-explanations or implementation details, Hebrew only.

EXAMPLES OF GOOD CODE:
- result = df['column_name'].sum()
- result = df.groupby('category')['value'].mean()
- result = df[df['status'] == 'active'].shape[0]
- result = df.nlargest(5, 'amount')[['name', 'amount']]

EXAMPLES OF BAD CODE (DON'T DO THIS):
- print(df['column'].mean())  # Don't print, assign to result
- def calculate(): return df['col'].mean()  # Don't just define function
- df['column'].mean()  # Don't leave hanging expression

Use this format:

Question: the input question you must answer
Thought: think about what to do
Action: the action to take (use tool name exactly)
Action Input: the input to the action
Observation: the result of the action
... (repeat Thought/Action/Action Input/Observation as needed)
Thought: I now know the final answer
Final Answer: the final answer to the original question

Question: {input}

{agent_scratchpad}"""
        
        prompt = PromptTemplate(
            input_variables=["input", "tools", "tool_names", "agent_scratchpad"],
            template=template
        )
        
        return create_react_agent(
            llm=self.llm,
            tools=tools,
            prompt=prompt
        )
    
    def _load_file(self, file_path: str) -> str:
        """Load file and return preview of first 10 rows."""
        state = self._get_thread_state()
        try:
            # Detect file type and load
            if file_path.lower().endswith(('.xlsx', '.xls')):
                state['current_df'] = pd.read_excel(file_path)
            else:
                state['current_df'] = pd.read_csv(file_path)
            
            # Create preview (first 10 rows only)
            preview_df = state['current_df'].head(10)
            
            state['file_preview'] = {
                "total_rows": len(state['current_df']),
                "total_columns": len(state['current_df'].columns),
                "columns": list(state['current_df'].columns),
                "dtypes": {col: str(dtype) for col, dtype in state['current_df'].dtypes.items()},
                "sample_data": preview_df.to_dict(orient='records'),
                "shape": state['current_df'].shape
            }
            
            return json.dumps(state['file_preview'], indent=2, ensure_ascii=False)
        
        except Exception as e:
            return f"Error loading file: {str(e)}\n{traceback.format_exc()}"
    
    def _execute_code(self, code: str) -> str:
        """Execute Python code with the loaded DataFrame."""
        state = self._get_thread_state()
        
        if state['current_df'] is None:
            return "Error: No file loaded. Use 'load_file' first."
        
        # Clean the code
        code = code.strip()
        if code.startswith("```python"):
            code = code.replace("```python", "").replace("```", "").strip()
        elif code.startswith("```"):
            code = code.replace("```", "").strip()
        
        # Create safe namespace
        namespace = {
            "df": state['current_df'],
            "pd": pd,
            "np": np
        }
        
        try:
            # Execute code
            exec(code, namespace)
            
            # Check for result variable
            if "result" not in namespace:
                error_msg = "Error: Code must assign final answer to 'result' variable"
                state['execution_history'].append({
                    "code": code,
                    "error": error_msg,
                    "success": False
                })
                return error_msg
            
            result = namespace["result"]
            
            # Format result
            formatted_result = self._format_result(result)
            
            # Record success
            state['execution_history'].append({
                "code": code,
                "result": formatted_result,
                "success": True
            })
            
            return f"Success! Result:\n{formatted_result}"
        
        except Exception as e:
            error_msg = f"Execution Error: {str(e)}\n{traceback.format_exc()}"
            state['execution_history'].append({
                "code": code,
                "error": error_msg,
                "success": False
            })
            return error_msg
    
    def _format_result(self, result: Any) -> str:
        """Format result for display."""
        if isinstance(result, pd.DataFrame):
            return result.to_string()
        elif isinstance(result, pd.Series):
            return result.to_string()
        elif isinstance(result, (np.integer, np.floating)):
            return str(result.item())
        elif isinstance(result, np.ndarray):
            return str(result.tolist())
        else:
            return str(result)
    
    def _get_history(self, _: str = "") -> str:
        """Get execution history."""
        state = self._get_thread_state()
        
        if not state['execution_history']:
            return "No execution history yet."
        
        history_str = "Execution History:\n"
        for i, entry in enumerate(state['execution_history'], 1):
            history_str += f"\n--- Attempt {i} ---\n"
            history_str += f"Code:\n{entry['code']}\n"
            if entry['success']:
                history_str += f"Result: {entry.get('result', 'N/A')}\n"
            else:
                history_str += f"Error: {entry.get('error', 'N/A')}\n"
        
        return history_str
    
    def analyze(self, file_path: str, question: str) -> Dict[str, Any]:
        """
        Analyze a file and answer a question.
        Thread-safe for parallel requests.
        
        Args:
            file_path: Path to CSV or Excel file
            question: Natural language question about the data
            
        Returns:
            Dictionary with result, code, and metadata
        """
        # Reset thread-local state
        self._reset_thread_state()
        state = self._get_thread_state()
        
        # Create executor for this request
        executor = self._create_executor()
        
        # Create full query
        full_query = f"""File: {file_path}
Question: {question}

Start by loading the file to understand its structure, then write clean Python code to answer the question."""
        
        try:
            # Execute agent
            result = executor.invoke({"input": full_query})
            
            # Get successful execution
            successful_execution = None
            for entry in reversed(state['execution_history']):
                if entry.get('success'):
                    successful_execution = entry
                    break
            
            return {
                "status": "success",
                "answer": result.get("output", "No answer generated"),
                "executed_code": successful_execution.get('code') if successful_execution else None,
                "result": successful_execution.get('result') if successful_execution else None,
                "attempts": len(state['execution_history']),
                "file_info": state['file_preview']
            }
        
        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "traceback": traceback.format_exc(),
                "attempts": len(state['execution_history']),
                "execution_history": state['execution_history']
            }


# Convenience function for quick usage
def analyze_file(
    file_path: str,
    question: str,
    vllm_endpoint: str,
    model_deployment: str,
    **kwargs
) -> Dict[str, Any]:
    """
    Quick function to analyze a file with a question.
    
    Args:
        file_path: Path to CSV or Excel file
        question: Question to answer
        vllm_endpoint: vLLM endpoint URL
        model_deployment: Model deployment name
        **kwargs: Additional arguments for DataAnalysisAgent
        
    Returns:
        Analysis results
    """
    agent = DataAnalysisAgent(
        vllm_endpoint=vllm_endpoint,
        model_deployment=model_deployment,
        **kwargs
    )
    return agent.analyze(file_path, question)

