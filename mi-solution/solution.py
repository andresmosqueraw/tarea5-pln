# ---- Modelos a evaluar (4B en 4-bit) ----
MODELS = [
    "unsloth/mistral-7b-instruct-v0.3-bnb-4bit",
    "unsloth/gemma-2-9b-it-bnb-4bit",
]

NEWS_LABELS = [
"alt.atheism",
"comp.graphics",
"comp.os.ms-windows.misc",
"comp.sys.ibm.pc.hardware",
"comp.sys.mac.hardware",
"comp.windows.x",
"misc.forsale",
"rec.autos",
"rec.motorcycles",
"rec.sport.baseball",
"rec.sport.hockey",
"sci.crypt",
"sci.electronics",
"sci.med",
"sci.space",
"soc.religion.christian",
"talk.politics.guns",
"talk.politics.mideast",
"talk.politics.misc",
"talk.religion.misc"
]

NEWS_PROMPTS = {

1: """You are a strict 20 Newsgroups classifier.
IGNORE headers / noise before classifying:
- lines starting with: From:, Subject:, Organization:, Lines:, Reply-To:, NNTP, Keywords:
- email addresses, URLs, signatures
- quoted replies (lines starting with '>' or 'In article')

Decide ONLY the MAIN topical category of the remaining body text.
No chain-of-thought. No <think>. No explanation. No JSON. No quotes.

Allowed labels (verbatim):
{labels}

TEXT:
{texto}

Output EXACTLY one label from the above list:""",

2: """20 Newsgroups disambiguation — strict mode.

Rules for common confusions (but DO NOT explain your reasoning in output):
• soc.religion.christian = Christian doctrine/worship
• alt.atheism = arguments about belief/nonbelief
• talk.politics.* = laws/policy/current events (guns → guns; Middle East → mideast; else → misc)
• rec.sport.* = sports (baseball/hockey)
• rec.motorcycles = riding/gear/maint
• rec.autos = cars/maintenance/buying
• comp.sys.ibm.pc.hardware = PC hardware
• comp.sys.mac.hardware = Mac hardware
• comp.graphics = graphics/rendering
• comp.windows.x = X11
• comp.os.ms-windows.misc = Windows not X11
• sci.electronics = circuitry/EE
• sci.crypt = crypto/security
• sci.med = medicine/healthcare
• sci.space = astronomy/spaceflight
• misc.forsale = classifieds

Ignore headers/signatures/quotes. Pick exactly ONE label.
No chain-of-thought. No <think>. No explanation. No JSON.

Allowed labels (verbatim):
{labels}

TEXT:
{texto}

Output = the label ONLY:""",

3: """20 Newsgroups — JSON output.

Texts can be long emails (median ~176 words).
Focus ONLY on the dominant recurring theme across paragraphs; ignore tangents, headers, signatures, quoted replies.

No chain-of-thought. No <think>. No commentary.

Return JSON with exactly ONE valid label from:
{labels}

TEXT:
{texto}

Output ONLY:
{{"label": ""}}"""
}


SENTIMENT_PROMPTS = {
    1: """You are a sentiment classifier for a MULTI-DOMAIN reviews dataset.

Allowed sentiment labels (verbatim):
positive
negative

Allowed domains (verbatim):
books, dvd, kitchen, electronics

RULES:
• ignore email noise (addresses, signatures)
• NO chain-of-thought
• NO <think>
• NO explanations
• output exactly the label
• one single token output ONLY

TEXT:
{texto}

Answer ONLY the sentiment:""",


    2: """You classify PRODUCT REVIEWS.

Domains vocabulary guidance:
• books = reading experience, plot, author, chapters
• dvd   = movie, actors, video, picture quality, scenes
• kitchen = appliances, food prep, cooking usage
• electronics = devices, gadgets, hardware

Sentiment rules:
• positive = overall endorsement / value / quality
• negative = complaint / disappointment / failure

Allowed labels:
positive
negative

Do NOT explain.
Do NOT output JSON.
Output only the label.

TEXT:
{texto}

Label:""",


    3: """Classify sentiment of this review.

Return exactly one of:
["positive","negative"]

TEXT:
{texto}

Return JSON ONLY in the form:
{{"label": "<one>"}}"""
}


import re, json, torch, pandas as pd
from typing import List, Dict, Optional
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm.auto import tqdm
from sklearn.metrics import *
from IPython.display import display

# ======================================================================
# CONFIG GLOBAL
# ======================================================================
NEWS_TEST = "/home/estudiante/punto2/mi-solution/news/test.csv"
GOLD_COL  = "label_name"
TEXT_COL  = "text"

GEN_KW = dict(max_new_tokens=24, do_sample=False)   # determinista
SMOKE_LIMIT = 3

HEADER_PREFIXES = ("from:", "subject:", "organization:", "reply-to:", "lines:", "nntp", "keywords:")

# ======================================================================
# strip / parser
# ======================================================================
def strip_headers_quotes(x: str, max_chars: int = 1500) -> str:
    lines = []
    for ln in x.splitlines():
        l = ln.strip()
        if not l:
            continue
        ll = l.lower()
        if ll.startswith(HEADER_PREFIXES) or ll.startswith(">") or ll.startswith("in article"):
            continue
        lines.append(l)
    body = "\n".join(lines)
    return body[:max_chars]     # cabeza del cuerpo (NO cola)

def extract_label(raw: str, allowed_labels: List[str]) -> Optional[str]:
    if raw is None:
        return None
    s = raw.strip()
    allowed_set = set(allowed_labels)

    # JSON directo
    try:
        j = json.loads(s)
        if isinstance(j, dict):
            for k in ("label","Label","category","Category"):
                if k in j:
                    cand = str(j[k]).strip()
                    if cand in allowed_set:
                        return cand
    except:
        pass

    # tokens
    s_clean = re.sub(r'[\r\t]', ' ', s).strip()
    tokens = re.split(r'[\s,;:|"“”\'`]+', s_clean)

    # exact case
    for t in tokens:
        if t in allowed_set:
            return t

    # case insensitive
    lower_map = {lbl.lower(): lbl for lbl in allowed_labels}
    for t in tokens:
        tt = t.lower()
        if tt in lower_map:
            return lower_map[tt]

    # substring full string — última chance
    low = s_clean.lower()
    for low_lbl, canon in lower_map.items():
        if low_lbl in low:
            return canon
    return None

# ======================================================================
# MODELO
# ======================================================================
def load_chat_model(model_name: str):
    torch.set_grad_enabled(False)
    tok = AutoTokenizer.from_pretrained(model_name)
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto" if torch.cuda.is_available() else None,
            torch_dtype=(getattr(torch,"bfloat16",None) or getattr(torch,"float16",None))
        )
    except ValueError as e:
        if "GPU RAM" in str(e) or "device_map" in str(e) or "CPU or the disk" in str(e):
            print(f"[WARNING] Model too large for GPU, trying CPU offload...")
            try:
                # Intentar con llm_int8_enable_fp32_cpu_offload
                model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    device_map="auto",
                    torch_dtype=(getattr(torch,"bfloat16",None) or getattr(torch,"float16",None)),
                    llm_int8_enable_fp32_cpu_offload=True
                )
            except (TypeError, ValueError) as e2:
                # Si el modelo no soporta llm_int8_enable_fp32_cpu_offload, intentar sin ese parámetro
                print(f"[WARNING] Model doesn't support CPU offload parameter, trying alternative loading...")
                try:
                    # Intentar con device_map="balanced" o "balanced_low_0"
                    model = AutoModelForCausalLM.from_pretrained(
                        model_name,
                        device_map="balanced",
                        torch_dtype=(getattr(torch,"bfloat16",None) or getattr(torch,"float16",None))
                    )
                except Exception as e3:
                    # Último intento: cargar sin device_map específico
                    print(f"[WARNING] Trying to load without specific device_map...")
                    model = AutoModelForCausalLM.from_pretrained(
                        model_name,
                        torch_dtype=(getattr(torch,"bfloat16",None) or getattr(torch,"float16",None))
                    )
        else:
            raise
    model.eval()
    return tok, model

@torch.no_grad()
def generate_label_for_text(tokenizer, model, text: str,
                            pid: int, prompts: Dict[int,str], allowed_labels: List[str]) -> Optional[str]:
    # prompt principal
    prompt = prompts[pid].format(texto=text, labels="\n".join(allowed_labels))
    print("\n==== PROMPT USED ====\n", prompt[:300],"\n====================\n")
    msgs = [{"role":"user","content":prompt}]
    inputs = tokenizer.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True,
                                           return_dict=True, return_tensors="pt",
                                           enable_thinking=False)  # Desactivar thinking/chain-of-thought
    inputs = {k:v.to(model.device) for k,v in inputs.items()}

    out = model.generate(**inputs, **GEN_KW,
                         num_beams=1,use_cache=True,pad_token_id=tokenizer.eos_token_id)
    gen = tokenizer.decode(out[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
    # HARD KILL COT OUTPUT  ---------------------------
    gen = gen.replace("<think>", "").replace("</think>", "")
    # remove any newline chatter
    gen = gen.strip().split("\n")[0].strip()
    if gen.startswith("<think>"):
        gen = gen.split("</think>")[-1].strip()
    print(f"[GEN-RAW] {repr(gen[:120])}")

    lab = extract_label(gen, allowed_labels)
    if lab is not None:
        return lab

    # FALLBACK JSON
    sp = (
      'Return a JSON {"label": "<one>"} from this set:\n'
      f'{allowed_labels}\n\nText:\n{text[:800]}\n\n'
      'Output only: {"label":"..."}'
    )
    msgs2=[{"role":"user","content":sp}]
    inp2 = tokenizer.apply_chat_template(msgs2, add_generation_prompt=True, tokenize=True,
                                         return_dict=True, return_tensors="pt")
    inp2={k:v.to(model.device) for k,v in inp2.items()}
    out2 = model.generate(**inp2, max_new_tokens=24, do_sample=False,
                          num_beams=1,use_cache=True,pad_token_id=tokenizer.eos_token_id)
    gen2 = tokenizer.decode(out2[0][inp2["input_ids"].shape[-1]:], skip_special_tokens=True)
    print(f"[GEN-FB] {repr(gen2[:120])}")
    return extract_label(gen2, allowed_labels)


# ======================================================================
# METRICAS
# ======================================================================
def compute_metrics(df: pd.DataFrame, labels: List[str]) -> dict:
    y_true=df["gold"].astype(str).tolist()
    y_pred=df["pred"].astype(str).tolist()
    acc=accuracy_score(y_true,y_pred)
    p_macro,r_macro,f1_macro,_ = precision_recall_fscore_support(y_true,y_pred,labels=labels,average="macro",zero_division=0)
    p_micro,r_micro,f1_micro,_ = precision_recall_fscore_support(y_true,y_pred,labels=labels,average="micro",zero_division=0)
    return dict(accuracy=acc,f1_macro=f1_macro,f1_micro=f1_micro,
                precision_macro=p_macro,recall_macro=r_macro)


# ======================================================================
# INFERENCIA NEWS
# ======================================================================
summary_rows=[]
for model_name in MODELS:
    print("\n"+"#"*100)
    print(f"[LOAD] {model_name}")
    try:
        tok,model = load_chat_model(model_name)
    except Exception as e:
        print(f"[ERROR] Failed to load {model_name}: {e}")
        print(f"[SKIP] Skipping {model_name} and continuing with next model...")
        continue

    df=pd.read_csv(NEWS_TEST)
    if SMOKE_LIMIT:
        df=df.head(SMOKE_LIMIT)

    for pid in [1,2,3]:
        print("\n"+"-"*100)
        print(f"[PROMPT {pid}]")
        print(NEWS_PROMPTS[pid][:200],"...\n")

        preds=[]
        for i in tqdm(range(len(df)),desc=f"{model_name}|p{pid}"):
            raw=str(df.loc[i,TEXT_COL])
            clean=strip_headers_quotes(raw,max_chars=1500)
            pred=generate_label_for_text(tok,model,clean,pid,NEWS_PROMPTS,NEWS_LABELS)
            preds.append(pred if pred else "")

        res=pd.DataFrame(dict(
            text=df[TEXT_COL],
            gold=df[GOLD_COL].astype(str),
            pred=preds))
        res["correct"]=(res.gold==res.pred).astype(int)

        print("\n=== EJEMPLOS ===")
        for j in range(len(res)):
            print(f"[{j}] gold={res.iloc[j].gold} pred={res.iloc[j].pred}")
            print(res.iloc[j].text[:300],"...\n")

        m = compute_metrics(res,NEWS_LABELS)
        print("\n==== sanity check labels ====")
        for i,l in enumerate(NEWS_LABELS):
            print(i, l)
        print("N =", len(NEWS_LABELS))
        summary_rows.append([model_name,pid,m["accuracy"],m["f1_macro"],m["f1_micro"],
                             m["precision_macro"],m["recall_macro"]])
    
    # Liberar memoria del modelo antes de cargar el siguiente
    del model, tok
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

# ======================================================================
# INFERENCIA MULTI (sentiment 2-clases con 4 dominios)
# ======================================================================

MULTI_TEST = "/home/estudiante/punto2/mi-solution/mdsd/test.csv"
MULTI_LABELS = ["positive","negative"]
MULTI_TEXT_COL = "text"
MULTI_GOLD_COL = "label"       # en multi la columna se llama "label"

for model_name in MODELS:
    print("\n"+"#"*100)
    print(f"[LOAD MULTI] {model_name}")
    try:
        tok,model = load_chat_model(model_name)
    except Exception as e:
        print(f"[ERROR] Failed to load {model_name}: {e}")
        print(f"[SKIP] Skipping {model_name} and continuing with next model...")
        continue

    df=pd.read_csv(MULTI_TEST)
    if SMOKE_LIMIT:
        df=df.head(SMOKE_LIMIT)

    for pid in [1,2,3]:
        print("\n"+"-"*100)
        print(f"[MULTI PROMPT {pid}]")
        print(SENTIMENT_PROMPTS[pid][:200],"...\n")

        preds=[]
        for i in tqdm(range(len(df)),desc=f"{model_name}|multi-p{pid}"):
            raw=str(df.loc[i,MULTI_TEXT_COL])
            clean = raw[:1500]   # para evitar ultra largos
            pred=generate_label_for_text(tok,model,clean,pid,SENTIMENT_PROMPTS,MULTI_LABELS)
            preds.append(pred if pred else "")

        res=pd.DataFrame(dict(
            text=df[MULTI_TEXT_COL],
            gold=df[MULTI_GOLD_COL].astype(str),
            pred=preds))
        res["correct"]=(res.gold==res.pred).astype(int)

        print("\n=== MULTI EJEMPLOS ===")
        for j in range(len(res)):
            print(f"[{j}] gold={res.iloc[j].gold} pred={res.iloc[j].pred}")
            print(res.iloc[j].text[:300],"...\n")

        m = compute_metrics(res,MULTI_LABELS)
        summary_rows.append([f"{model_name}-MULTI",pid,m["accuracy"],m["f1_macro"],m["f1_micro"],
                             m["precision_macro"],m["recall_macro"]])
    
    # Liberar memoria del modelo antes de cargar el siguiente
    del model, tok
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

# ======================================================================
# TABLA FINAL CON RESULTADOS DE AMBOS DATASETS
# ======================================================================
summary_df=pd.DataFrame(summary_rows,columns=["model","prompt","accuracy","f1_macro","f1_micro","precision_macro","recall_macro"])

print("\n"+"="*100)
print("RESULT SUMMARY (Newsgroups + Multi-Domain Sentiment):")
print("="*100)
display(summary_df.sort_values(["model","f1_macro"],ascending=[True,False]))