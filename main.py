import io
import os
import re
import json
import fitz          # PyMuPDF
import pdfplumber
import pytesseract
import pandas as pd

from fastapi import FastAPI, UploadFile, File, Response, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from openai import OpenAI

# ---------- Config ----------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("Set OPENAI_API_KEY env var")
client = OpenAI(api_key=OPENAI_API_KEY)

app = FastAPI(title="Extractor Estado de Resultados → Excel")
app.mount("/",
          StaticFiles(directory="static", html=True),
          name="static")

TARGET_TOKENS = [
    "estado de resultados",
    "estado de resultados integrales",
    "estado de resultados por función",
    "estado de resultados por naturaleza",
]

# JSON Schema for Structured Outputs (fields in Spanish)
JSON_SCHEMA = {
    "name": "EstadoResultados",
    "strict": True,
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "empresa": {"type": "string"},
            "periodos": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "etiqueta_periodo": {"type": "string"},   # p.ej. "31-12-2024"
                        "moneda": {"type": "string"},             # p.ej. "CLP", "USD"
                        "unidad": {"type": "string"},             # p.ej. "miles", "millones"
                        "ingresos": {"type": "number"},
                        "costo_de_ventas": {"type": "number"},
                        "utilidad_bruta": {"type": "number"},
                        "gastos_operacionales": {"type": "number"},
                        "resultado_operacional": {"type": "number"},
                        "otros_ingresos_gastos": {"type": "number"},
                        "resultado_antes_impuestos": {"type": "number"},
                        "impuesto_a_las_ganancias": {"type": "number"},
                        "utilidad_neta": {"type": "number"},
                        "ebitda": {"type": "number"},
                        "depreciacion_amortizacion": {"type": "number"}
                    },
                    "required": ["etiqueta_periodo", "moneda", "unidad", "utilidad_neta"]
                }
            },
            "observaciones": {"type": "string"}
        },
        "required": ["periodos"]
    }
}

PROMPT_INSTRUCTIONS = """
Eres un analista contable. A partir del texto suministrado (páginas del PDF) extrae SOLO el
“Estado de Resultados” (o “Estado de Resultados Integrales”).

Reglas:
- Devuelve números en formato numérico (no en texto), y normaliza todos según la unidad detectada
  (p.ej., si el estado dice “en millones de CLP”, convierte a millones).
- Si algún rubro no aparece explícito, puedes inferirlo cuando sea trivial
  (p.ej., utilidad bruta = ingresos - costo de ventas). Si no es posible, omítelo.
- Usa etiquetas de periodo tal como aparezcan (ej.: “31-12-2024”, “12M 2024”).
- Moneda y unidad deben venir de los encabezados (CLP, UF, USD; miles/millones).
- Si hay estados por naturaleza vs función, prioriza la versión con mayor detalle.
- No inventes, no cambies de moneda, no conviertas UF↔CLP: solo reporta lo que esté en el documento.
- Salida EXACTAMENTE conforme al esquema JSON que te doy.
"""

def find_candidate_pages(pdf_bytes: bytes) -> list[int]:
    """Quick text scan to locate pages with target keywords."""
    pages = []
    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        for i, page in enumerate(pdf.pages):
            try:
                text = (page.extract_text() or "").lower()
            except Exception:
                text = ""
            if any(tok in text for tok in TARGET_TOKENS):
                pages.append(i)
    # fallback: if nothing found, scan all (but cap at first 6 to be safe)
    return pages if pages else list(range(0, min(6, len(pdf.pages))))

def page_to_text(pdf_bytes: bytes, page_num: int) -> str:
    """Extract text; if empty (scan), OCR via Tesseract."""
    text = ""
    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        try:
            text = pdf.pages[page_num].extract_text() or ""
        except Exception:
            text = ""

    if text.strip():
        return text

    # OCR path: render page to image then Tesseract (Spanish/English)
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    page = doc.load_page(page_num)
    pix = page.get_pixmap(dpi=300)
    img_bytes = pix.tobytes("png")
    ocr = pytesseract.image_to_string(
        io.BytesIO(img_bytes),
        lang="spa+eng"
    )
    return ocr or ""

def extract_text_block(pdf_bytes: bytes) -> str:
    pages = find_candidate_pages(pdf_bytes)
    blocks = []
    for p in pages:
        t = page_to_text(pdf_bytes, p)
        if t.strip():
            blocks.append(f"--- Página {p+1} ---\n{t}")
    return "\n\n".join(blocks)

def call_openai_structured(extracted_text: str) -> dict:
    """
    Use OpenAI with Structured Outputs (JSON Schema) to normalize the Income Statement.
    We use chat.completions for wide compatibility with SDKs.
    Docs: Structured Outputs & API reference.
    """
    # IMPORTANT: model name—pick a fast vision-capable or text-only (we already OCRed)
    model = "gpt-4o-mini"  # good cost/quality tradeoff

    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": PROMPT_INSTRUCTIONS.strip()},
            {"role": "user", "content": extracted_text[:200000]}  # safety cap
        ],
        response_format={ "type": "json_schema", "json_schema": JSON_SCHEMA }
    )
    raw = resp.choices[0].message.content
    # Some SDKs return JSON directly; others return string—parse both cases:
    if isinstance(raw, str):
        try:
            return json.loads(raw)
        except Exception:
            # strip code fences if any
            cleaned = re.sub(r"^```json|```$", "", raw.strip(), flags=re.MULTILINE)
            return json.loads(cleaned)
    return raw

def to_excel_bytes(payload: dict, raw_text: str) -> bytes:
    # Flatten 'periodos'
    rows = []
    for p in payload.get("periodos", []):
        row = {
            "etiqueta_periodo": p.get("etiqueta_periodo"),
            "moneda": p.get("moneda"),
            "unidad": p.get("unidad"),
            "ingresos": p.get("ingresos"),
            "costo_de_ventas": p.get("costo_de_ventas"),
            "utilidad_bruta": p.get("utilidad_bruta"),
            "gastos_operacionales": p.get("gastos_operacionales"),
            "resultado_operacional": p.get("resultado_operacional"),
            "otros_ingresos_gastos": p.get("otros_ingresos_gastos"),
            "resultado_antes_impuestos": p.get("resultado_antes_impuestos"),
            "impuesto_a_las_ganancias": p.get("impuesto_a_las_ganancias"),
            "utilidad_neta": p.get("utilidad_neta"),
            "ebitda": p.get("ebitda"),
            "depreciacion_amortizacion": p.get("depreciacion_amortizacion"),
        }
        rows.append(row)
    df = pd.DataFrame(rows)

    out = io.BytesIO()
    with pd.ExcelWriter(out, engine="xlsxwriter") as writer:
        df.to_excel(writer, index=False, sheet_name="Estado_Resultados")
        # Metadata
        meta = {
            "empresa": [payload.get("empresa")],
            "observaciones_modelo": [payload.get("observaciones")],
            "paginas_capturadas": [raw_text[:5000]],  # include first chunk for traceability
        }
        pd.DataFrame(meta).to_excel(writer, index=False, sheet_name="Metadata")
    out.seek(0)
    return out.read()

@app.post("/api/extract")
async def extract_endpoint(file: UploadFile = File(...)):
    if file.content_type not in ("application/pdf", "application/x-pdf"):
        raise HTTPException(400, "Debe subir un PDF.")
    pdf_bytes = await file.read()
    if len(pdf_bytes) > 40 * 1024 * 1024:
        raise HTTPException(400, "PDF demasiado grande (máx. ~40MB demo).")

    text_block = extract_text_block(pdf_bytes)
    if not text_block.strip():
        raise HTTPException(422, "No pude extraer texto u OCR del PDF.")

    result = call_openai_structured(text_block)
    excel = to_excel_bytes(result, text_block)

    headers = {
        "Content-Disposition": 'attachment; filename="estado_resultados.xlsx"'
    }
    return Response(content=excel, headers=headers,
                    media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
