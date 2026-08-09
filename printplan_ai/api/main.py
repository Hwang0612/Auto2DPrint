"""FastAPI backend for PrintPlan AI."""
from __future__ import annotations
import tempfile
from pathlib import Path
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from printplan_ai import __version__
from printplan_ai.config import M1Params, M2Params, M3Params, M4Params
from printplan_ai.models import Stage1Output, Stage2Output, Stage3Output
from printplan_ai.stages import parse_pdf, reconstruct, generate_toolpath, synthesise_gcode

app = FastAPI(title="PrintPlan AI", version=__version__)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True,
                   allow_methods=["*"], allow_headers=["*"])

@app.get("/api/health")
def health(): return {"status": "ok", "version": __version__}

@app.post("/api/m1/parse", response_model=Stage1Output)
async def m1_parse(file: UploadFile = File(...), drawing_scale: float = Form(50.0),
                   wall_layer_pattern: str = Form(r"wall")) -> Stage1Output:
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(400, "Upload a .pdf file.")
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(await file.read()); tmp_path = Path(tmp.name)
    try:
        return parse_pdf(tmp_path, M1Params(drawing_scale=drawing_scale,
                                            wall_layer_pattern=wall_layer_pattern))
    finally: tmp_path.unlink(missing_ok=True)

@app.post("/api/m2/reconstruct", response_model=Stage2Output)
def m2_reconstruct(stage1: Stage1Output, params: M2Params | None = None) -> Stage2Output:
    return reconstruct(stage1, params)

@app.post("/api/m3/toolpath", response_model=Stage3Output)
def m3_toolpath(stage2: Stage2Output, params: M3Params | None = None,
                n_layers: int = 10) -> Stage3Output:
    return generate_toolpath(stage2, params, n_layers=n_layers)

@app.post("/api/m4/gcode")
def m4_gcode(stage3: Stage3Output, params: M4Params | None = None):
    """Generate G-code from a Stage-3 toolpath."""
    from fastapi.responses import PlainTextResponse
    gcode = synthesise_gcode(stage3, params)
    return PlainTextResponse(gcode, media_type="text/plain")

@app.post("/api/m5/validate")
def m5_stub(): raise HTTPException(501, "M5 not yet implemented.")
