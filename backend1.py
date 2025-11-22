import os
import faiss
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS
from langchain_ollama import ChatOllama

# ------------------------------
# CONFIG VARIABLES (DEFINE FIRST!)
# ------------------------------
INDEX_DIR = "uc_act_index"
FILE_PATH = "ukpga_20250022_en.pdf"

# ------------------------------
# Step 1: Load and split document
# ------------------------------
def load_and_split(file_path):
    loader = PyPDFLoader(file_path)
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150,
        separators=["\n\n", "\n", ".", " "]
    )

    chunks = splitter.split_documents(docs)
    return chunks

# ------------------------------
# Step 2: Build FAISS Vector Store
# ------------------------------
def build_vector_store(chunks, embeddings, index_path=INDEX_DIR):
    dim = len(embeddings.embed_query("hello"))
    index = faiss.IndexFlatL2(dim)

    vector_store = FAISS(
        embedding_function=embeddings,
        index=index,
        docstore=InMemoryDocstore(),
        index_to_docstore_id={}
    )

    vector_store.add_documents(chunks)
    vector_store.save_local(index_path)
    return vector_store

# ------------------------------
# Step 3: Summarization (Task 2)
# ------------------------------
def summarize_act(vector_store, model_name="qwen3:1.7b"):
    results = vector_store.similarity_search("overview of the entire act", k=6)
    context = "\n\n".join(doc.page_content for doc in results)

    prompt = f"""
Summarize the Universal Credit Act 2025 in exactly 5–10 bullet points.
Focus only on:
- Purpose
- Key definitions
- Eligibility
- Obligations
- Enforcement elements

Context:
{context}

Return only bullet points.
"""

    model = ChatOllama(model=model_name)
    response = model.invoke(prompt)
    return response.content


#----------------------
#Task3
#-----------------------

import json

# -----------------------------------------
# Task 3: Extraction Helper (RAG + JSON)
# -----------------------------------------
def extract_section(vector_store, query, model_name="qwen3:1.7b"):
    """
    Generic extractor using RAG and JSON-only output.
    """
    # Retrieve top chunks for accuracy
    docs = vector_store.similarity_search(query, k=6)
    context = "\n\n".join(d.page_content for d in docs)

    prompt = f"""
You are an AI assistant extracting legal information from the Universal Credit Act 2025.

Context:
{context}

Task:
{query}

Rules:
- Return ONLY valid JSON.
- Do NOT add explanation.
- Do NOT include commentary.
- JSON must be a list of strings.

JSON Output:
"""

    model = ChatOllama(model=model_name)
    response = model.invoke(prompt).content

    # Safely load JSON (removes text if LLM adds extra)
    try:
        return json.loads(response)
    except:
        # fallback: extract JSON using simple heuristic
        try:
            start = response.index('[')
            end = response.rindex(']') + 1
            return json.loads(response[start:end])
        except:
            return []

# -----------------------------------------
# Task 3: Category-Specific Extractors
# -----------------------------------------
def extract_definitions(vs):
    return extract_section(
        vs,
        "Extract ALL legal definitions from the Act. Return a JSON list of strings."
    )

def extract_eligibility(vs):
    return extract_section(
        vs,
        "Extract ALL eligibility rules for Universal Credit and LCW/LCWRA from the Act. Return JSON list."
    )

def extract_payments(vs):
    return extract_section(
        vs,
        "Extract ALL rules about payment amounts, calculations, annual adjustments, LCWRA, protected amounts, and ESA IR payments. Return JSON list."
    )

def extract_obligations(vs):
    return extract_section(
        vs,
        "Extract ALL obligations or responsibilities described in the Act (Secretary of State duties, regulatory duties, etc.). Return JSON list."
    )

def extract_enforcement(vs):
    return extract_section(
        vs,
        "Extract ALL enforcement or regulatory powers mentioned in the Act. If none exist, return an empty JSON list. Return JSON list."
    )

def extract_record_keeping(vs):
    return extract_section(
        vs,
        "Extract any record-keeping, reporting, or calculation requirements (e.g., CPI determination, annual rates publication). Return JSON list."
    )

# -----------------------------------------
# Task 3: Final JSON Builder
# -----------------------------------------
def build_task3_json(vector_store):
    result = {
        "definitions": extract_definitions(vector_store),
        "eligibility": extract_eligibility(vector_store),
        "payments": extract_payments(vector_store),
        "obligations": extract_obligations(vector_store),
        "enforcement": extract_enforcement(vector_store),
        "record_keeping": extract_record_keeping(vector_store),
    }
    return result


#----------------
#Task4
#----------------

import json
import re
from typing import List, Dict

# ------------------------------
# Task 4: Rule Checker
# ------------------------------

# The 6 rules from the assignment
RULES = [
    "Act must define key terms",
    "Act must specify eligibility criteria",
    "Act must specify responsibilities of the administering authority",
    "Act must include enforcement or penalties",
    "Act must include payment calculation or entitlement structure",
    "Act must include record-keeping or reporting requirements"
]

def call_llm_for_rule(context: str, rule_text: str, model_name="qwen3:1.7b") -> str:
    """
    Call the Ollama model with a strict JSON-only instruction for a single rule.
    Returns raw model text.
    """
    prompt = f"""
You are an assistant verifying whether a legal rule is satisfied by the provided context.
Do NOT produce any extra text outside of JSON.

Context (evidence chunks):
{context}

Task:
For the rule: "{rule_text}", analyze the context and produce a JSON object with these keys:
- rule: the rule text (string)
- status: one of "PASS", "FAIL", or "INCONCLUSIVE" (string)
- evidence: a short quoted excerpt from the Act that supports the decision. If no relevant text, use "No matching section found".
- confidence: a decimal number between 0 and 1 indicating confidence in the judgment (e.g., 0.95).

Rules for your answer:
1. Return ONLY valid JSON (no commentary, no explanation).
2. Keep evidence short (one or two sentences or a specific quoted clause).
3. If multiple supporting excerpts exist, include the most direct one.
4. If the Act explicitly states the requirement, prefer PASS with confidence >= 0.8.
5. If the Act is silent, return FAIL with confidence <= 0.4.
6. If the info is ambiguous or partial, return INCONCLUSIVE with a confidence around 0.5-0.7.

Output JSON example:
{{
  "rule": "Act must define key terms",
  "status": "PASS",
  "evidence": "Section 1: 'the standard allowance means ...'",
  "confidence": 0.92
}}
"""

    model = ChatOllama(model=model_name)
    response = model.invoke(prompt)
    return response.content

def extract_json_from_text(text: str):
    """
    Try to extract JSON from model output robustly.
    Returns parsed JSON object or None.
    """
    # Quick try: direct parse
    try:
        return json.loads(text)
    except Exception:
        pass

    # Heuristic: find first '{' and last '}' and parse substring
    try:
        start = text.index('{')
        end = text.rindex('}') + 1
        candidate = text[start:end]
        return json.loads(candidate)
    except Exception:
        pass

    # Heuristic: find a JSON array/object pattern using regex
    json_candidates = re.findall(r'(\{(?:.|\s)*\})', text)
    for cand in json_candidates:
        try:
            return json.loads(cand)
        except Exception:
            continue

    return None

def check_single_rule(vector_store, rule_text: str, k: int = 6, model_name="qwen3:1.7b") -> Dict:
    """
    Retrieve evidence with RAG, call LLM to evaluate the rule, and return a dict result.
    """
    # Retrieve top-k chunks
    docs = vector_store.similarity_search(rule_text, k=k)
    if not docs:
        # No chunks found at all
        return {
            "rule": rule_text,
            "status": "FAIL",
            "evidence": "No matching section found",
            "confidence": 0.25
        }

    # Build context string with chunk metadata if available
    context_pieces = []
    for i, d in enumerate(docs):
        meta = getattr(d, "metadata", {}) or {}
        # include a short prefix for traceability
        prefix = ""
        if "source" in meta:
            prefix = f"[source: {meta.get('source')}] "
        context_pieces.append(f"{prefix}{d.page_content.strip()[:1000]}")  # limit chunk length in prompt
    context = "\n\n---\n\n".join(context_pieces)

    raw = call_llm_for_rule(context, rule_text, model_name=model_name)
    parsed = extract_json_from_text(raw)

    if parsed is None:
        # fallback: create a conservative response using retrieval only
        top_excerpt = docs[0].page_content.strip()[:400].replace("\n", " ")
        return {
            "rule": rule_text,
            "status": "INCONCLUSIVE",
            "evidence": top_excerpt if top_excerpt else "No matching section found",
            "confidence": 0.50
        }

    # Validate fields and normalize
    rule_field = parsed.get("rule", rule_text)
    status_field = parsed.get("status", "INCONCLUSIVE")
    evidence_field = parsed.get("evidence", "")
    conf_field = parsed.get("confidence", 0.5)

    # Ensure status casing and allowed values
    status_field = status_field.upper() if isinstance(status_field, str) else "INCONCLUSIVE"
    if status_field not in ("PASS", "FAIL", "INCONCLUSIVE"):
        status_field = "INCONCLUSIVE"

    # Clamp confidence to [0,1] and ensure numeric
    try:
        conf_field = float(conf_field)
    except Exception:
        conf_field = 0.5
    conf_field = max(0.0, min(1.0, conf_field))

    return {
        "rule": rule_field,
        "status": status_field,
        "evidence": evidence_field,
        "confidence": round(conf_field, 3)
    }

def run_task4(vector_store, model_name="qwen3:1.7b"):
    print("Running Task 4: Rule Checks...")
    results = []
    for r in RULES:
        print(f"\nChecking rule: {r}")
        res = check_single_rule(vector_store, r, k=6, model_name=model_name)
        print("->", res["status"], f"(confidence: {res['confidence']})")
        results.append(res)

    # Save to JSON file
    with open("task4_output.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)

    print("\nTask 4 complete. Results saved to task4_output.json")
    return results




# ------------------------------
# MAIN FLOW
# ------------------------------
if __name__ == "__main__":
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-mpnet-base-v2"
    )

    if os.path.exists(INDEX_DIR):
        print("Loading existing FAISS index...")
        vector_store = FAISS.load_local(
            INDEX_DIR, embeddings, allow_dangerous_deserialization=True
        )
    else:
        print("Building FAISS index...")
        chunks = load_and_split(FILE_PATH)
        vector_store = build_vector_store(chunks, embeddings)

    summary = summarize_act(vector_store)
    print("\n=== SUMMARY OUTPUT ===\n")
    print(summary)

    print("\n=== Running Task 3 Extraction ===\n")
    task3_output = build_task3_json(vector_store)
    print(json.dumps(task3_output, indent=4))

    with open("task3_output.json", "w", encoding="utf-8") as f:
        json.dump(task3_output, f, indent=4)

    print("\n=== Running Task 4: Rule Checks ===\n")
    task4_output = run_task4(vector_store)
    print(json.dumps(task4_output, indent=4))

    with open("task4_output.json", "w", encoding="utf-8") as f:
        json.dump(task4_output, f, indent=4)
