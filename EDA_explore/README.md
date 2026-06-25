# 🧠 Local Data Analysis Agent

A **local-first ChatGPT-like Data Analysis Agent** powered by **Ollama + Gemma + Pandas Code Interpreter**.

This project enables you to:

* Upload CSV / Excel files
* Ask natural language questions
* Automatically generate and execute pandas code
* Visualize results (tables, charts)
* Persist multi-session conversations

---

## 🚀 Features

### 🧠 LLM-powered Code Interpreter

* Uses **Gemma (via Ollama)** to generate Python (pandas) code
* Automatically executes code and returns results
* Supports:

  * Aggregations
  * GroupBy analysis
  * Filtering
  * Statistical summaries

---

### 📊 Data Analysis & Visualization

* Supports:

  * CSV / Excel upload
  * DataFrame preview
  * Chart generation (matplotlib)
* Output types:

  * Table (DataFrame / Series)
  * Numeric results
  * Charts

---

### 💬 Conversational Interface

* Built with **Streamlit Chat UI**
* Multi-turn conversation support
* Context-aware responses (limited history window)

---

### 💾 Session Memory

* Save & load sessions
* Resume analysis anytime
* Stored locally (JSON)

---

### 🔁 Self-healing Execution (Basic)

* Automatically retries when code execution fails
* Uses LLM to fix errors

---

## 🏗️ Architecture

```text
User (Streamlit UI)
        ↓
      app.py
        ↓
     agent.py
   ┌───────────────┐
   ↓               ↓
llm.py         executor.py
 (Gemma)        (sandbox)
   ↓               ↓
   └──→ Result ←───┘
           ↓
      memory.py
```

---

## 📂 Project Structure

```
data_agent/
│
├── app.py              # Streamlit UI (entry point)
├── agent.py            # Core agent logic
├── llm.py              # LLM (Ollama) wrapper
├── executor.py         # Python code execution (sandbox)
├── data_manager.py     # DataFrame management
├── memory.py           # Session persistence
│
├── config.md           # Model & prompt configuration
├── config_loader.py    # Config parser
│
├── sessions/           # Stored conversations
└── uploads/            # Uploaded datasets
```

---

## ⚙️ Installation

### 1. Install dependencies

```bash
pip install streamlit pandas matplotlib openpyxl ollama
```

---

### 2. Install & run Ollama

Install Ollama:
👉 https://ollama.ai

Run model:

```bash
ollama run gemma:2b
```

---

### 3. Start the app

```bash
streamlit run app.py
```

---

## 🧪 Usage

### Step 1 — Upload Data

* Upload `.csv` or `.xlsx`

### Step 2 — Ask Questions

Examples:

#### 📊 Analysis

* "What is the average sales?"
* "Group by region and calculate total revenue"
* "Show top 5 products by sales"

#### 📈 Visualization

* "Plot sales distribution"
* "Create a bar chart of sales by region"

---

## ⚙️ Configuration

All model & prompt settings are stored in:

```
config.md
```

Example:

```
# MODEL
gemma:2b

# SYSTEM_PROMPT
(controls how the model generates pandas code)
```

👉 You can modify prompts without touching Python code.

---

## 🔒 Security Note

The project uses a restricted execution environment:

```python
exec(code, {"__builtins__": {}}, local_vars)
```

However, this is **NOT fully secure**.

For production use, consider:

* Docker sandbox
* RestrictedPython
* Isolated execution service

---

## ⚠️ Current Limitations

* No full schema awareness (column guessing may fail)
* Limited multi-step reasoning
* Basic error recovery (single retry)
* Not fully secure sandbox

---

## 🛣️ Roadmap

### 🔥 v2 (Recommended Next Step)

* Schema injection (df.columns, df.dtypes)
* Better prompt engineering
* Improved chart generation
* Smarter retry loop

---

### 🤖 v3 (Agent Upgrade)

* ReAct-style reasoning
* Tool calling system
* Multi-dataset support

---

### 🏢 v4 (Production-grade)

* LangGraph architecture
* Plugin system
* Multi-agent workflows

---

## 🎯 Why This Project Matters

This project demonstrates:

* LLM Tool Usage (Code Generation + Execution)
* Data Analysis Automation
* Local-first AI system design
* Agent-based architecture

👉 Suitable for:

* AI Engineer portfolio
* Technical PM / TPM showcase
* Data + LLM integration projects

---

## 📜 License

MIT License
