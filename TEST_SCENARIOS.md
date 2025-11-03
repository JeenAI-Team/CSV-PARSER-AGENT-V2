# Agent.py Test Scenarios

## ✅ Should Work Well:

### 1. Simple Aggregations
- "What is the total sales?" → `result = df['sales'].sum()`
- "How many rows?" → `result = len(df)`
- "Average price?" → `result = df['price'].mean()`

### 2. Grouping with Context
- "Sales by quarter?" → `result = df.groupby('quarter')['sales'].sum()`
- "Top 5 customers?" → `result = df.nlargest(5, 'sales')[['name', 'sales']]`
- "Max sales?" → `result = df.loc[df['sales'].idxmax()]`

### 3. Canvas/Charts
- "Create pie chart" → `result = df.groupby('category')['value'].sum().reset_index()`
- "Line chart of sales" → `result = df[['date', 'sales']].sort_values('date')`
- Returns JSON array for Canvas

---

## ⚠️ May Have Issues:

### 1. Complex Multi-Step Analysis
- Multiple calculations in sequence
- **Issue:** Gemma may hallucinate Observations between steps
- **Mitigation:** Truncation cuts hallucinated Observations

### 2. Conversation with Multiple Files
- Upload file1 → ask question → upload file2 → ask question
- **Issue:** Gemma may get confused by old file context
- **Mitigation:** Agent extracts last user question only

### 3. Hebrew Column Names
- Columns like: "מכירות", "רבעון"
- **Should work:** Agent handles Hebrew in JSON
- **Risk:** Encoding issues in some edge cases

---

## ❌ Known Limitations:

### 1. Gemma Not Trained on ReAct
- Will sometimes write Observations itself
- Will sometimes skip Action/Action Input format
- **Solution:** Strong prompts + truncation help but not 100%

### 2. File Prompts Not Updated in DB
- DB still has: "Use This CSV Analysis In Your Answer:"
- Should have: "...use chart_generator tool to visualize"
- **Impact:** Main LLM may not create chart_generator tool call
- **Solution:** Need to UPDATE database or reseed

### 3. Canvas Data Format
- Agent returns: `result` as JSON string
- Backend needs to: parse and pass to main LLM
- Main LLM needs to: create chart_generator tool call
- **Risk:** Format mismatches at any step

---

## 🔧 Recommended Next Steps:

### Priority 1: Fix file_prompts in Database
```sql
UPDATE playground_properties 
SET file_prompts = '{"csv_parser_prompt": "CSV Analysis Result:", "xlsx_parser_prompt": "Excel Data - If user asked for chart/canvas, use chart_generator tool:"}'::jsonb
WHERE id = 1;
```

### Priority 2: Test Full Canvas Flow
1. Upload Excel with sales data
2. Ask: "תן לי pie chart של מכירות לפי רבעון"
3. Check logs for:
   - `[Excel] Returning RESULT field` ✓
   - Main LLM receives JSON data ✓
   - Main LLM creates `chart_generator` tool call ✓
   - Canvas displays chart ✓

### Priority 3: Monitor Gemma Hallucinations
- Check logs for `[WARNING] Model hallucinated Observation`
- If frequent → may need stronger prompts or different model

---

## 📊 Overall Assessment:

**Current State: 70% Ready**

**Pros:**
- ✅ Core agent logic works
- ✅ Can analyze most Excel files
- ✅ Returns correct data format
- ✅ Handles Hebrew
- ✅ Thread-safe

**Cons:**
- ⚠️ Gemma hallucinations need monitoring
- ⚠️ file_prompts not updated in DB
- ⚠️ Canvas flow depends on proper prompt

**Recommendation:**
- For production: Consider using GPT-4/Claude (better ReAct support)
- For Gemma: Current setup works but needs monitoring
- Update DB prompts immediately for full Canvas support

