"""
Script de Evaluación de Clasificación de Texto usando Modelos de Lenguaje Basados en Decodificadores.

Este script evalúa múltiples modelos de lenguaje (Mistral, Gemma) en dos tareas de clasificación:
1. Clasificación de 20 Newsgroups (20 clases)
2. Clasificación de Sentimiento Multi-Dominio (2 clases: positivo/negativo)

Para cada tarea, prueba 10 plantillas de prompts diferentes y genera:
- Métricas de rendimiento detalladas (accuracy, F1, precision, recall)
- Resultados agregados promediados entre modelos
- Matrices de confusión para los prompts con mejor rendimiento
- Tiempo total de inferencia

Autor: Generado para evaluación de clasificación NLP
"""

from typing import List, Dict, Optional, Tuple, Any, Union

# ---- Modelos a evaluar (4B en 4-bit) ----
MODELS: List[str] = [
    "unsloth/mistral-7b-instruct-v0.3-bnb-4bit",
    "unsloth/gemma-2-9b-it-bnb-4bit",
]
"""Lista de identificadores de modelos de HuggingFace a evaluar."""

NEWS_LABELS: List[str] = [
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

NEWS_PROMPTS: Dict[int, str] = {

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

Output EXACTLY one label from the above list. NOTHING ELSE.""",

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

Output ONLY the single label.""",

3: """20 Newsgroups — JSON output.

Texts can be long emails (median ~176 words).
Focus ONLY on the dominant recurring theme across paragraphs; ignore tangents, headers, signatures, quoted replies.

No chain-of-thought. No <think>. No commentary.

Return JSON with exactly ONE valid label from:
{labels}

TEXT:
{texto}

Output ONLY:
{{"label": ""}}""",

# --- 4) Few-shot (in-context learning) ---
4: """Classify 20 Newsgroups topic. Output exactly one label string from the allowed set. No explanations.

Examples:
Text: "Driver issues after installing the new card; IRQ conflicts on my 486."
Label: comp.sys.ibm.pc.hardware

Text: "Is justification by faith alone biblical?"
Label: soc.religion.christian

Text: "Where can I find torque specs for the GS500?"
Label: rec.motorcycles

Text: "Looking to sell a used Pentium motherboard."
Label: misc.forsale

ALLOWED:
{labels}

Now classify:
TEXT:
{texto}

Answer with EXACTLY one label from the list.""",

# --- 5) Gamified framing ---
5: """You are the Gatekeeper of Newsgroup City. Present the correct district badge (the label) for the message.

Districts (labels):
{labels}

Rules:
• Ignore headers, signatures, and quoted replies.
• Choose the single best-matching district for the core topic.
• Speak ONLY the badge name (the label). Nothing else.

TEXT:
{texto}

Badge:""",

# --- 6) Rubric / checklist (decision rules) ---
6: """Topic rubric (apply silently; do not print your reasoning):
(1) If the main topic is government/current affairs → talk.politics.* (guns/mideast/misc).
(2) If main topic is religion → soc.religion.christian (Christian) or alt.atheism (belief debates) or talk.religion.misc (other).
(3) If technical computing:
    • hardware issues PC/Mac → comp.sys.ibm.pc.hardware / comp.sys.mac.hardware
    • X11 → comp.windows.x
    • Windows (non-X11) → comp.os.ms-windows.misc
    • graphics/rendering → comp.graphics
(4) Science/tech:
    • electronics/EE → sci.electronics
    • cryptography/security → sci.crypt
    • medicine/health → sci.med
    • space/astronomy → sci.space
(5) Recreation:
    • cars → rec.autos
    • motorcycles → rec.motorcycles
    • baseball/hockey → rec.sport.baseball / rec.sport.hockey
(6) Classifieds → misc.forsale

Allowed labels:
{labels}

TEXT:
{texto}

Output EXACTLY one label string. NOTHING ELSE.""",

# --- 7) Contrastive (closest vs. second-closest, but output only label) ---
7: """Pick the SINGLE best 20 Newsgroups label by internally comparing the two most likely categories and choosing the closer one. Do not reveal comparisons.

Allowed labels:
{labels}

Ignore headers/signatures/quotes. Focus on the dominant theme.

TEXT:
{texto}

Output ONLY the chosen label.""",

# --- 8) Persona / role (strict editor) ---
8: """Act as the Editor-in-Chief for the 20 Newsgroups archive.
Your job: assign each message to exactly ONE section.

Policy:
• Remove headers, footers, signatures, quoted replies from consideration.
• Base the decision on the central topic of the body.
• No explanations, no JSON.

Sections (labels):
{labels}

TEXT:
{texto}

Return EXACTLY one section label from the list.""",

# --- 9) Keyword-guided / domain lexicon (heuristic guidance) ---
9: """Use the following cue lexicon silently (not rules, just hints):
Politics: "bill, congress, election, policy, rights, gun, israel, palestine, mideast"
Religion: "church, bible, christian, atheist, faith, doctrine"
Computing: "IRQ, driver, BIOS, SCSI, PCI, kernel, Xlib, X11, NT, Win3.1, DirectX, render"
Science: "circuit, voltage, capacitor, crypto, cipher, vaccine, diagnosis, telescope, shuttle"
Recreation: "batting, pitcher, puck, goalie, torque, carburetor, sedan"
Marketplace: "for sale, WTS, selling, price, shipped, OBO"

Allowed labels:
{labels}

TEXT:
{texto}

Output EXACTLY one matching label. Nothing else.""",

# --- 10) Calibrated/argmax internal scoring (but output only label) ---
10: """Internally score the text against all labels by topical fit; choose the argmax. Do NOT reveal scores or reasoning.

Valid labels:
{labels}

Ignore headers/signatures/quotes. Use ONLY the core body.

TEXT:
{texto}

Output EXACTLY one label from the list, and nothing else."""
}

SENTIMENT_PROMPTS: Dict[int, str] = {
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
{{"label": "<one>"}}""",

    # --- Few-shot (in-context learning) ---
    4: """You classify review sentiment for: books, dvd, kitchen, electronics.
Output exactly one token: positive or negative. No explanations.

Examples:
Review: "Great characters and pacing kept me hooked." (domain=books)
Label: positive

Review: "The blender leaked after one use." (domain=kitchen)
Label: negative

Review: "Crisp picture, the director’s cut shines." (domain=dvd)
Label: positive

Review: "Battery dies within hours, terrible build." (domain=electronics)
Label: negative

Now classify:
TEXT:
{texto}

Label:""",

    # --- Gamified prompt (same task, playful framing) ---
    5: """You are the Gatekeeper of Reviewland.
Rule: If the review praises usefulness/quality/value → say 'positive'.
If it reports defects/disappointment/poor value → say 'negative'.
Speak only the password (the label). No story, no explanation.

TEXT:
{texto}

Password:""",

    # --- Rubric/checklist (decision rules without revealing thoughts) ---
    6: """Decide review sentiment using this silent checklist:
(1) Outcome: success/enjoyment vs failure/disappointment.
(2) Product quality: durable/effective vs broken/defective.
(3) Value: worth buying vs waste of money.
Apply negation correctly (e.g., "not bad" = mild positive).
Output exactly: positive OR negative. No extra text.

TEXT:
{texto}

Answer:""",

    # --- Contrastive cues (pros vs cons) ---
    7: """Classify sentiment of a product review.
Privately weigh pros vs cons; if pros dominate → positive, else → negative.
Handle irony and negation.
Output ONLY one token:
positive
negative

TEXT:
{texto}

Label:""",

    # --- Persona/role prompt (strict grader) ---
    8: """Act as a strict SENTIMENT JUDGE for consumer reviews (domains: books, dvd, kitchen, electronics).
Penalize reports of defects, returns, refunds, broken items, shipping damage.
Reward comments about reliability, enjoyment, and value for money.
Output exactly one token (positive|negative). Do not explain.

TEXT:
{texto}

Verdict:""",

    # --- Keyword-guided + domain lexicon (heuristic guidance) ---
    9: """Binary sentiment classification for multi-domain reviews.

Heuristic guidance (not rules):
Positive cues: excellent, love, works, reliable, recommend, worth, enjoyable
Negative cues: broke, refund, return, defective, poor, hate, disappointing, waste
Books/dvd: story/acting/pacing/cinematography
Kitchen/electronics: build/battery/warranty/performance

Output exactly one:
positive
negative

TEXT:
{texto}

Label:""",

    # --- Calibrated scoring → threshold to label (but output only label) ---
    10: """Score the review internally from 1 (very negative) to 5 (very positive) considering quality, performance, and value.
Map score ≤2 → negative; ≥4 → positive; 3 → use overall tone to break tie.
Do NOT reveal the score. Output ONLY the label token.

TEXT:
{texto}

Label:"""
}


import re, json, torch, pandas as pd, time
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM, PreTrainedModel
from tqdm.auto import tqdm
from sklearn.metrics import *
from IPython.display import display

NEWS_TEST: str = "/home/estudiante/punto2/mi-solution/news/test.csv"
"""Ruta al archivo CSV del dataset de prueba de 20 Newsgroups."""

MULTI_TEST: str = "/home/estudiante/punto2/mi-solution/mdsd/test.csv"
"""Ruta al archivo CSV del dataset de prueba de Sentimiento Multi-Dominio."""

# ======================================================================
# CONFIG GLOBAL
# ======================================================================
GOLD_COL: str = "label_name"
"""Nombre de la columna para las etiquetas verdaderas en el dataset de Newsgroups."""

TEXT_COL: str = "text"
"""Nombre de la columna para el texto de entrada en los datasets."""

GEN_KW: Dict[str, Union[int, bool]] = dict(max_new_tokens=24, do_sample=False)
"""Parámetros de generación para la inferencia del modelo. Determinista (sin muestreo)."""

SMOKE_LIMIT: Optional[int] = 1
"""Limita el número de ejemplos de prueba para pruebas rápidas. Establecer a None para usar el conjunto de prueba completo."""

HEADER_PREFIXES: Tuple[str, ...] = ("from:", "subject:", "organization:", "reply-to:", "lines:", "nntp", "keywords:")
"""Prefijos de encabezados de email para filtrar al limpiar el texto."""

# ======================================================================
# strip / parser
# ======================================================================
def strip_headers_quotes(x: str, max_chars: int = 1500) -> str:
    """
    Elimina encabezados de email, respuestas citadas y firmas del texto.
    
    Filtra líneas que comienzan con encabezados comunes de email (From:, Subject:, etc.),
    respuestas citadas (líneas que comienzan con '>' o 'In article'), y líneas vacías.
    Retorna el texto del cuerpo limpio truncado a max_chars caracteres.
    
    Args:
        x: Cadena de texto de entrada (típicamente un mensaje de email/newsgroup)
        max_chars: Número máximo de caracteres a retornar (por defecto: 1500)
    
    Returns:
        Cadena de texto limpia con encabezados y citas eliminadas, truncada a max_chars
    """
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
    """
    Extrae una etiqueta válida de la salida cruda del modelo.
    
    Intenta múltiples estrategias para extraer una etiqueta válida:
    1. Parsear formato JSON (busca claves "label", "Label", "category", "Category")
    2. Coincidencia exacta sensible a mayúsculas en tokens
    3. Coincidencia sin distinguir mayúsculas en tokens
    4. Coincidencia de subcadena en la cadena completa (último recurso)
    
    Args:
        raw: Cadena de salida cruda del modelo
        allowed_labels: Lista de cadenas de etiquetas válidas para comparar
    
    Returns:
        Cadena de etiqueta coincidente si se encuentra, None en caso contrario
    """
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
def load_chat_model(model_name: str) -> Tuple[AutoTokenizer, PreTrainedModel]:
    """
    Carga un modelo de lenguaje pre-entrenado y su tokenizador.
    
    Intenta múltiples estrategias de carga para manejar diferentes restricciones de memoria GPU:
    1. Carga estándar con device_map="auto"
    2. Descarga a CPU con llm_int8_enable_fp32_cpu_offload
    3. Mapeo de dispositivos balanceado
    4. Respaldo a carga solo en CPU
    
    Args:
        model_name: Identificador del modelo de HuggingFace (ej: "unsloth/mistral-7b-instruct-v0.3-bnb-4bit")
    
    Returns:
        Tupla de (tokenizer, model) - ambos listos para inferencia
    
    Raises:
        Exception: Si todas las estrategias de carga fallan
    """
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
def generate_label_for_text(
    tokenizer: AutoTokenizer, 
    model: PreTrainedModel, 
    text: str,
    pid: int, 
    prompts: Dict[int, str], 
    allowed_labels: List[str]
) -> Optional[str]:
    """
    Genera una etiqueta de clasificación para un texto dado usando un modelo de lenguaje.
    
    Usa la plantilla de prompt especificada para generar una etiqueta. Si la generación inicial
    no produce una etiqueta válida, recurre a un prompt con formato JSON.
    
    Args:
        tokenizer: Tokenizador pre-entrenado para el modelo
        model: Modelo de lenguaje pre-entrenado para generación de texto
        text: Texto de entrada a clasificar
        pid: ID del prompt (clave en el diccionario de prompts)
        prompts: Diccionario que mapea IDs de prompts a plantillas de prompts
        allowed_labels: Lista de cadenas de etiquetas válidas
    
    Returns:
        Cadena de etiqueta extraída si es exitoso, None en caso contrario
    """
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
def compute_metrics(df: pd.DataFrame, labels: List[str]) -> Dict[str, float]:
    """
    Calcula métricas de clasificación a partir de predicciones y valores reales.
    
    Calcula accuracy, precisión, recall y puntajes F1 promediados macro/micro.
    
    Args:
        df: DataFrame con columnas "gold" (etiquetas verdaderas) y "pred" (etiquetas predichas)
        labels: Lista de todas las cadenas de etiquetas posibles (para promedio macro)
    
    Returns:
        Diccionario que contiene:
            - accuracy: Accuracy general
            - f1_macro: Puntaje F1 promediado macro
            - f1_micro: Puntaje F1 promediado micro
            - precision_macro: Precisión promediada macro
            - recall_macro: Recall promediado macro
    """
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
print("\n"+"="*100)
print("STARTING INFERENCE - Timer started")
print("="*100)
start_time = time.time()

summary_rows: List[List[Union[str, int, float]]] = []
"""Lista de filas de resultados: [model_name, prompt_id, accuracy, f1_macro, f1_micro, precision_macro, recall_macro]"""

# Almacenar resultados para matrices de confusión: {dataset: {prompt: {model: (y_true, y_pred)}}}
confusion_data: Dict[str, Dict[int, Dict[str, Tuple[List[str], List[str]]]]] = {
    "Newsgroups": {}, 
    "Multi-Domain Sentiment": {}
}
"""Diccionario anidado que almacena tuplas (y_true, y_pred) para cada dataset, prompt y modelo."""
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
    if SMOKE_LIMIT is not None:
        df=df.head(SMOKE_LIMIT)

    for pid in range(1, 11):
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
        
        # Guardar resultados para matriz de confusión
        if pid not in confusion_data["Newsgroups"]:
            confusion_data["Newsgroups"][pid] = {}
        confusion_data["Newsgroups"][pid][model_name] = (res["gold"].tolist(), res["pred"].tolist())
    
    # Liberar memoria del modelo antes de cargar el siguiente
    del model, tok
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

# ======================================================================
# INFERENCIA MULTI (sentiment 2-clases con 4 dominios)
# ======================================================================

MULTI_LABELS: List[str] = ["positive","negative"]
"""Etiquetas de sentimiento válidas para el dataset de Sentimiento Multi-Dominio."""

MULTI_TEXT_COL: str = "text"
"""Nombre de la columna para el texto de entrada en el dataset de Sentimiento Multi-Dominio."""

MULTI_GOLD_COL: str = "label"
"""Nombre de la columna para las etiquetas verdaderas en el dataset de Sentimiento Multi-Dominio."""

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
    if SMOKE_LIMIT is not None:
        df=df.head(SMOKE_LIMIT)

    for pid in range(1, 11):
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
        
        # Guardar resultados para matriz de confusión
        if pid not in confusion_data["Multi-Domain Sentiment"]:
            confusion_data["Multi-Domain Sentiment"][pid] = {}
        confusion_data["Multi-Domain Sentiment"][pid][f"{model_name}-MULTI"] = (res["gold"].tolist(), res["pred"].tolist())
    
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

# ======================================================================
# TABLA AGRUPADA POR PROMPT (PROMEDIOS ENTRE MODELOS)
# ======================================================================
# Separar Newsgroups y MULTI
news_df = summary_df[~summary_df["model"].str.contains("-MULTI", na=False)].copy()
multi_df = summary_df[summary_df["model"].str.contains("-MULTI", na=False)].copy()

# Agrupar por prompt y calcular promedios para Newsgroups
news_grouped = news_df.groupby("prompt").agg({
    "accuracy": "mean",
    "f1_macro": "mean",
    "f1_micro": "mean",
    "precision_macro": "mean",
    "recall_macro": "mean"
}).reset_index()
news_grouped["dataset"] = "Newsgroups"

# Agrupar por prompt y calcular promedios para MULTI
multi_grouped = multi_df.groupby("prompt").agg({
    "accuracy": "mean",
    "f1_macro": "mean",
    "f1_micro": "mean",
    "precision_macro": "mean",
    "recall_macro": "mean"
}).reset_index()
multi_grouped["dataset"] = "Multi-Domain Sentiment"

# Combinar ambas tablas
grouped_summary = pd.concat([news_grouped, multi_grouped], ignore_index=True)
grouped_summary = grouped_summary[["dataset", "prompt", "accuracy", "f1_macro", "f1_micro", "precision_macro", "recall_macro"]]

print("\n"+"="*100)
print("PROMPT PERFORMANCE SUMMARY (Averaged across models):")
print("="*100)
display(grouped_summary.sort_values(["dataset", "f1_macro"], ascending=[True, False]))

# ======================================================================
# MATRICES DE CONFUSIÓN
# ======================================================================
print("\n"+"="*100)
print("CONFUSION MATRICES")
print("="*100)

# Generar matrices de confusión para el mejor prompt de cada dataset
for dataset_name in ["Newsgroups", "Multi-Domain Sentiment"]:
    dataset_df = grouped_summary[grouped_summary["dataset"] == dataset_name]
    if len(dataset_df) == 0:
        continue
    
    # Encontrar el mejor prompt (mayor f1_macro)
    best_prompt_row = dataset_df.loc[dataset_df["f1_macro"].idxmax()]
    best_prompt = int(best_prompt_row["prompt"])
    
    print(f"\n{'='*100}")
    print(f"Dataset: {dataset_name} - Best Prompt: {best_prompt} (F1-macro: {best_prompt_row['f1_macro']:.4f})")
    print(f"{'='*100}")
    
    if best_prompt not in confusion_data[dataset_name]:
        print(f"No data available for prompt {best_prompt}")
        continue
    
    # Combinar predicciones de todos los modelos para este prompt
    all_y_true = []
    all_y_pred = []
    for model_name, (y_true, y_pred) in confusion_data[dataset_name][best_prompt].items():
        all_y_true.extend(y_true)
        all_y_pred.extend(y_pred)
    
    # Obtener labels según el dataset
    if dataset_name == "Newsgroups":
        labels = NEWS_LABELS
    else:
        labels = MULTI_LABELS
    
    # Generar matriz de confusión
    cm = confusion_matrix(all_y_true, all_y_pred, labels=labels)
    
    # Crear visualización
    plt.figure(figsize=(12, 10) if dataset_name == "Newsgroups" else (8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=[l[:20] + '...' if len(l) > 20 else l for l in labels],
                yticklabels=[l[:20] + '...' if len(l) > 20 else l for l in labels],
                cbar_kws={'label': 'Count'})
    plt.title(f'Confusion Matrix - {dataset_name}\nPrompt {best_prompt} (Best F1-macro: {best_prompt_row["f1_macro"]:.4f})', 
              fontsize=14, fontweight='bold')
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()
    
    # También mostrar como tabla de texto
    print(f"\nConfusion Matrix (Prompt {best_prompt} - {dataset_name}):")
    print(f"Labels: {labels}")
    cm_df = pd.DataFrame(cm, index=labels, columns=labels)
    display(cm_df)
    
    # Calcular métricas por clase
    print(f"\nPer-class metrics:")
    for i, label in enumerate(labels):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        tn = cm.sum() - tp - fp - fn
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        print(f"  {label}: Precision={precision:.3f}, Recall={recall:.3f}, F1={f1:.3f}")

# ======================================================================
# TIEMPO TOTAL DE INFERENCIA
# ======================================================================
end_time = time.time()
total_time = end_time - start_time
hours = int(total_time // 3600)
minutes = int((total_time % 3600) // 60)
seconds = int(total_time % 60)
milliseconds = int((total_time % 1) * 1000)

print("\n"+"="*100)
print("INFERENCE TIME SUMMARY")
print("="*100)
print(f"Total inference time: {total_time:.2f} seconds")
print(f"Formatted: {hours}h {minutes}m {seconds}s {milliseconds}ms")
print("="*100)