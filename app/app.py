# main.py
from typing import List, Optional, Dict, Any, Union
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

# Import your LangChain-native service class from earlier
# from services.ai_summary_lc import ai_summary_service
from utils.ai_service import ai_summary_service  # <-- replace with real import


# ---------- Pydantic Schemas ----------

class EmailAddress(BaseModel):
    name: Optional[str] = None
    address: Optional[str] = None

class FromField(BaseModel):
    emailAddress: Optional[EmailAddress] = None

class Email(BaseModel):
    id: Optional[str] = None
    subject: Optional[str] = "No Subject"
    from_: Optional[Union[str, FromField]] = Field(None, alias="from")
    receivedDateTime: Optional[str] = None
    sentDateTime: Optional[str] = None
    bodyPreview: Optional[str] = None
    body: Optional[str] = None

class Attachment(BaseModel):
    emailId: Optional[str] = None
    name: Optional[str] = None
    size: Optional[int] = 0
    contentType: Optional[str] = None

class ParsedAttachment(BaseModel):
    name: Optional[str] = None
    contentType: Optional[str] = None
    size: Optional[int] = 0
    parsedContent: Optional[str] = None
    projectInfo: Optional[Dict[str, Any]] = None

class SummarizeRequest(BaseModel):
    project_name: str = ""
    page_name: str = "enquiry-organizer"  # <- REQUIRED by service method
    emails: List[Email]
    attachments: Optional[List[Attachment]] = None
    parsed_attachments: Optional[List[ParsedAttachment]] = None
    # code: int
    # status: str
    # message: str
    # invoice_data: dict
    # errors: list
class SummarizeResponse(BaseModel):
    code: int
    status: str
    message: str
    summary: str
    project_name: str
    email_count: int
    errors: list


# ---------- FastAPI App ----------

app = FastAPI(title="AI Services API", version="1.0.0")

# CORS (adjust origins as needed)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # tighten in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
def health():
    return {"status": "healthy", "service": "ai-services", "version": "1.0.0"}


# ---------- Endpoint: POST /summarize ----------

@app.post("/summarize", response_model=SummarizeResponse)
async def generate_summary(payload: SummarizeRequest):
    try:
        if not payload.emails:
            raise HTTPException(status_code=400, detail="No emails provided")

        # Convert Pydantic models to plain dicts for the service
        emails = [e.model_dump(by_alias=True) for e in payload.emails]
        atts = [a.model_dump() for a in (payload.attachments or [])]
        parsed = [p.model_dump() for p in (payload.parsed_attachments or [])]

        # Call LangChain service (run sync in a thread to avoid blocking)
        def _run():
            return ai_summary_service.generate_summary(
                project_name=payload.project_name,
                emails=emails,
                page_name=payload.page_name,           # <-- pass page_name in correct slot
                attachments=atts,
                parsed_attachments=parsed
            )

        summary: Optional[str] = await run_in_threadpool(_run)



        return SummarizeResponse(
            code = 1,
            status = "success",
            message = "Summary Generation Successful",
            summary=summary,
            project_name=payload.project_name,
            email_count=len(emails),
            errors = []
        )
    except Exception as e:
        return SummarizeResponse(
            code = 99,
            status = "Failed",
            message = "Summary Generation Failed",
            summary="",
            project_name=payload.project_name,
            email_count=len(emails),
            errors = [str(e)]
        )