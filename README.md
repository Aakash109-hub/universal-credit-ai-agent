# Universal Credit Act 2025 – AI Legal Analysis Agent

This project is an AI-powered pipeline that analyses the **Universal Credit Act 2025** using **LangChain**, **FAISS**, **HuggingFace embeddings**, and **Ollama** models.

It was built as part of the **NIYAMR 48-hour AI Intern Assignment**.

The system performs **four major tasks**:

---

## 📌 **Tasks Overview**

### **✅ Task 1 – Document Loading & Chunking**

* Loads the Act from PDF
* Splits it into meaningful text chunks
* Prepares data for retrieval

### **✅ Task 2 – Act Summarisation**

Generates a clear 5–10 bullet summary including:

* Purpose
* Key definitions
* Eligibility
* Obligations
* Enforcement elements

### **✅ Task 3 – Structured Section Extraction**

Uses RAG (Retrieval-Augmented Generation) to extract:

* Definitions
* Eligibility rules
* Payment structures
* LCWRA + ESA IR rules
* Obligations
* Enforcement items
* Record-keeping requirements

Outputs are saved in:

```
task3_output.json
```

### **✅ Task 4 – Rule Evaluation Engine**

Evaluates **six compliance rules**:

1. Act must define key terms
2. Act must specify the eligibility criteria
3. Act must specify authority's responsibilities
4. Act must include enforcement or penalties
5. Act must define payment and entitlement rules
6. Act must include record-keeping or reporting rules

For each rule, the system returns:

```json
{
  "rule": "...",
  "status": "PASS / FAIL / INCONCLUSIVE",
  "evidence": "Extracted from the Act",
  "confidence": 0.92
}
```

Outputs saved in:

```
task4_output.json
```

---

# 🧠 **Technology Stack**

* **Python**
* **LangChain** (RAG pipeline + document loaders)
* **HuggingFace Embeddings** (all-mpnet-base-v2)
* **FAISS Vector Store**
* **Ollama** (Qwen 1.7B model)
* **PyPDFLoader**

---

# 📦 **Files in This Repository**

```
backend1.py                 # Main Python script containing Tasks 1–4
ukpga_20250022_en.pdf       # Universal Credit Act 2025 (analysis document)
NIYAMR_48_Hour_Intern_Assignment.pdf  # Assignment PDF
task3_output.json           # Output of structured extraction
task4_output.json           # Output of rule evaluations
```

---

# 🚀 **How to Run the Project**

### 1. Install dependencies

```bash
pip install langchain langchain-community langchain-text-splitters langchain-huggingface langchain-ollama faiss-cpu sentence-transformers
```

### 2. Make sure Ollama is installed and running

```bash
ollama pull qwen3:1.7b
```

### 3. Run the pipeline

```bash
python backend1.py
```

The script automatically:

* Creates FAISS index
* Summarizes the Act
* Extracts legal sections
* Evaluates rules
* Saves JSON outputs

---

# 📁 **Output Files**

| File                | Description                                                         |
| ------------------- | ------------------------------------------------------------------- |
| `task3_output.json` | Extracted legal sections (definitions, eligibility, payments, etc.) |
| `task4_output.json` | Rule evaluation results with evidence & confidence                  |

---

# 🧩 **Summary of System Flow**

1. **Load PDF → Chunk Text**
2. **Build Vector Store with Embeddings**
3. **Retrieve Relevant Chunks**
4. **Generate Summaries and JSON Outputs**
5. **Evaluate Rules with PASS/FAIL Logic**

This creates an automated legal interpretation assistant for government policy documents.
