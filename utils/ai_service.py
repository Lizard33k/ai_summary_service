"""
AI Summary & Info Extraction (LangChain-native)
- No `_process_json_extraction`
- Uses structured output parsers for clean JSON → Python types
"""

import os
import re
from typing import Dict, List, Optional, Any

from pydantic import BaseModel, Field, validator
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser, PydanticOutputParser
from langchain_core.runnables import Runnable
from langchain_core.messages import HumanMessage

import os

# Get path to project root (one level up from utils)
project_root = os.path.dirname(os.path.dirname(__file__))

# Build path to secret file
key_file = os.path.join(project_root, "secret_keys", "secret_keys1.txt")

with open(key_file, "r") as f:
    api_key = f.read().strip()  


# ---------- Pydantic Schemas (Structured Outputs) ----------

class CompanyInfo(BaseModel):
    companyName: str = Field(..., description="Client organisation or main corresponding client name")


class ProjectInfo(BaseModel):
    site_area: Optional[float] = Field(None, description="Site area in m²")
    gross_floor_area: Optional[float] = Field(None, description="Gross floor area in m²")
    height: Optional[float] = Field(None, description="Height in meters")
    number_of_floors: Optional[int] = None
    number_of_units: Optional[int] = None
    type_of_project: List[str] = Field(default_factory=list, description="List like ['Residential','Commercial']")
    country: Optional[str] = None
    total_fee: Dict[str, Optional[float]] = Field(default_factory=dict, description="USD amounts keyed by service")
    complexity_of_scope_of_work: Optional[str] = None

    # Accept "Not found" and blank strings gracefully
    @validator("*", pre=True)
    def _normalize_not_found(cls, v):
        if v is None:
            return None
        if isinstance(v, str) and v.strip().lower() in {"not found", ""}:
            return None
        return v

    # Accept strings with units; strip numeric part if needed
    @validator("site_area", "gross_floor_area", "height", pre=True)
    def _to_float_if_str(cls, v):
        if isinstance(v, (int, float)) or v is None:
            return v
        nums = re.findall(r"[+-]?\d+(?:\.\d+)?", str(v))
        return float(nums[0]) if nums else None

    @validator("number_of_floors", "number_of_units", pre=True)
    def _to_int_if_str(cls, v):
        if isinstance(v, int) or v is None:
            return v
        nums = re.findall(r"\d+", str(v))
        return int(nums[0]) if nums else None

    @validator("type_of_project", pre=True)
    def _to_list_capitalized(cls, v):
        if v is None:
            return []
        if isinstance(v, str):
            parts = [p.strip() for p in v.split(",") if p.strip()]
        else:
            parts = list(v)
        return [p.capitalize() for p in parts]

    @validator("country", pre=True)
    def _cap_country(cls, v):
        if v is None:
            return None
        s = str(v).strip()
        return s[:1].upper() + s[1:] if s else None

    @validator("total_fee", pre=True)
    def _fees_to_float(cls, v):
        """
        Accepts:
          {"architecture":"100000 USD","interior_design":"Not found","landscaping":"85000"}
        Produces:
          {"architecture":100000.0,"interior_design":None,"landscaping":85000.0}
        """
        if not isinstance(v, dict):
            return {}
        out = {}
        for k, val in v.items():
            if val is None:
                out[k] = None
                continue
            if isinstance(val, (int, float)):
                out[k] = float(val)
                continue
            s = str(val).strip().lower()
            if s in {"not found", ""}:
                out[k] = None
                continue
            nums = re.findall(r"[+-]?\d+(?:\.\d+)?", s)
            out[k] = float(nums[0]) if nums else None
        return out


class AssignedProject(BaseModel):
    assigned_project_id: str


# ---------- Service Class (LangChain) ----------

class AISummaryServiceLC:
    """LangChain-native service for summarization, extraction, tagging, and categorization"""

    def __init__(self):
        # Uses OpenAI-compatible Chat model via LangChain
        # If you're on OpenRouter, set OPENAI_BASE_URL + OPENAI_API_KEY accordingly.
        # self.llm = ChatOpenAI(
        # openai_api_base="http://localhost:11434/v1/",
        # openai_api_key="12345678",
        # model_name="gemma3:4b")
        self.llm = ChatOpenAI(
        openai_api_key=api_key,
        model_name="gpt-4o"
        )
        # Parsers
        self.company_parser = PydanticOutputParser(pydantic_object=CompanyInfo)
        self.project_parser = PydanticOutputParser(pydantic_object=ProjectInfo)
        self.tags_parser = JsonOutputParser()          # expects pure JSON array
        self.assign_parser = PydanticOutputParser(pydantic_object=AssignedProject)

        # Prompts
        self.summary_prompt = PromptTemplate.from_template(
            """You are an AI assistant for architectural projects.
Project Name: {project_name}
Context:
{email_block}
Page Name: {page_name}

Write a clear 4–6 sentence summary of the email thread.
If this is not an architectural project, reply exactly:
"This email thread is not related to an architectural project."
"""
        )

        self.company_prompt = PromptTemplate(
            template=(
                "Extract ONLY the client organisation or main corresponding client name "
                "from the text below and return {format_instructions}.\n\n"
                "Text:\n{summary}"
            ),
            input_variables=["summary"],
            partial_variables={"format_instructions": self.company_parser.get_format_instructions()},
        )

        self.project_prompt = PromptTemplate(
            template=(
                "From the following summary, extract project details and return {format_instructions}.\n"
                "Fields:\n"
                "- site_area (m²), gross_floor_area (m²), height (m)\n"
                "- number_of_floors, number_of_units\n"
                "- type_of_project (list: residential, commercial, hospitality, etc.)\n"
                "- country (based on location)\n"
                "- total_fee (USD) keys: architecture, interior_design, landscaping (if present)\n"
                "- complexity_of_scope_of_work\n\n"
                "If something is missing, you may omit it or set it to null.\n\n"
                "Summary:\n{summary}"
            ),
            input_variables=["summary"],
            partial_variables={"format_instructions": self.project_parser.get_format_instructions()},
        )

        self.tags_prompt = PromptTemplate.from_template(
            """Generate 3–8 relevant tags (JSON array only) for this architectural project context.

Project: {project_name}
Content:
{content}

Guidelines:
- Cover type (residential/commercial/hospitality/etc.), services (architecture/interior/landscaping),
  characteristics (luxury/sustainable/renovation/new-build), location context, scale, timeline.
Return a JSON array, nothing else."""
        )

        self.assign_prompt = PromptTemplate(
            template=(
                "You categorize threads into projects.\n\n"
                "THREAD:\n"
                "- Thread ID: {thread_id}\n"
                "- Client: {client_name}\n"
                "- Project Name: {project_name}\n"
                "- Tags: {tags}\n"
                "- Summary: {summary}\n\n"
                "AVAILABLE PROJECTS:\n{available_projects}\n\n"
                "EXISTING CLIENT PROJECTS:\n{existing_client_projects}\n\n"
                "RULES:\n"
                "1) Prefer existing client projects if tags/semantics match.\n"
                "2) Use tags and names/descriptions for best match.\n"
                "3) Suggest a new (available) project only if no good existing match.\n"
                "4) If none match, choose the first available.\n\n"
                "Return {format_instructions} only."
            ),
            input_variables=[
                "thread_id", "client_name", "tags", "summary", "project_name",
                "available_projects", "existing_client_projects"
            ],
            partial_variables={"format_instructions": self.assign_parser.get_format_instructions()},
        )

    # ---------- Public Methods (mirrors your original API surface) ----------

    def generate_summary(
        self,
        project_name: str,
        emails: List[Dict],
        page_name: str,
        attachments: Optional[List[Dict]] = None,
        parsed_attachments: Optional[List[Dict]] = None,
    ) -> Optional[str]:
        if not emails:
            return None
        email_block = self._prepare_email_block(emails, attachments, parsed_attachments)
        chain: Runnable = self.summary_prompt | self.llm
        result = chain.invoke({"project_name": project_name, "email_block": email_block, "page_name": page_name})
        text = result.content if hasattr(result, "content") else str(result)
        return text
### The main probem here is that the summary which is used to extract the company name might not always have the company name
### because the summary does not exclusively tell to include the company name
    # def extract_company_name(self, project_name: str, summary: str) -> Optional[Dict]:
    #     if not summary:
    #         return None
    #     chain: Runnable = self.company_prompt | self.llm | self.company_parser
    #     parsed: CompanyInfo = chain.invoke({"summary": summary})
#        return parsed.dict()

    # def extract_project_info(self, project_name: str, summary: str) -> Optional[Dict]:
    #     if not summary:
    #         return None
    #     chain: Runnable = self.project_prompt | self.llm | self.project_parser
    #     parsed: ProjectInfo = chain.invoke({"summary": summary})
    #     return parsed.dict()

    # def generate_tags(
    #     self,
    #     thread_id: str,
    #     summary: str,
    #     project_name: str,
    #     emails: Optional[List[Dict]] = None,
    # ) -> Optional[List[str]]:
    #     if not thread_id:
    #         return None
    #     content = summary.strip() if summary and summary.strip() and summary.strip() != "No summary available." else ""
    #     if not content and emails:
    #         content = self._emails_as_text(emails)

    #     chain: Runnable = self.tags_prompt | self.llm | self.tags_parser
    #     try:
    #         tags = chain.invoke({"project_name": project_name, "content": content})
    #         if isinstance(tags, list):
    #             cleaned = list({str(t).strip().lower() for t in tags if str(t).strip()})
    #             return cleaned[:8]
    #     except Exception:
    #         pass
    #     return ["architecture", "project", "inquiry"]

    # def categorize_with_tags(
    #     self,
    #     thread_id: str,
    #     client_name: str,
    #     tags: List[str],
    #     summary: str,
    #     project_name: str,
    #     available_projects: List[Dict],
    #     existing_client_projects: List[str],
    #     client_project_associations: Dict[str, List[str]],
    # ) -> Optional[str]:
    #     if not thread_id or not available_projects:
    #         return None

    #     # Keep inputs small enough for context
    #     ap = available_projects
    #     if len(ap) > 20:
    #         ap = ap[:20]
    #     ecp = existing_client_projects[:20]

    #     chain: Runnable = self.assign_prompt | self.llm | self.assign_parser
    #     out: AssignedProject = chain.invoke({
    #         "thread_id": thread_id,
    #         "client_name": client_name,
    #         "tags": ", ".join(tags) if tags else "None",
    #         "summary": summary[:500],
    #         "project_name": project_name,
    #         "available_projects": ap,
    #         "existing_client_projects": ecp
    #     })
    #     return out.assigned_project_id

    # ---------- Helpers ----------

    def _prepare_email_block(
        self,
        emails: List[Dict],
        attachments: Optional[List[Dict]] = None,
        parsed_attachments: Optional[List[Dict]] = None,
    ) -> str:
        lines = []
        for i, email in enumerate(emails, 1):
            lines.append(f"Email {i}:")
            lines.append(f"Subject: {email.get('subject', 'No Subject')}")
            from_info = email.get("from", "Unknown")
            if isinstance(from_info, dict):
                from_name = from_info.get("emailAddress", {}).get("name", "Unknown")
            else:
                from_name = str(from_info)
            lines.append(f"From: {from_name}")
            lines.append(f"Date: {email.get('receivedDateTime', email.get('sentDateTime', 'Unknown'))}")
            lines.append(f"Content: {email.get('bodyPreview', email.get('body', 'No content preview'))}")

            # Match attachments to this email
            if attachments:
                e_atts = [a for a in attachments if a.get("emailId") == email.get("id")]
                if e_atts:
                    lines.append("Attachments:")
                    for a in e_atts:
                        lines.append(f"  - {a.get('name','Unknown')} ({a.get('size',0)} bytes, {a.get('contentType','Unknown')})")
            lines.append("-" * 40)

        if parsed_attachments:
            lines.append("\n" + "=" * 50)
            lines.append("PARSED ATTACHMENT CONTENT")
            lines.append("=" * 50)
            for i, att in enumerate(parsed_attachments, 1):
                lines.append(f"\nAttachment {i}: {att.get('name','Unknown')}")
                lines.append(f"Type: {att.get('contentType','Unknown')}")
                lines.append(f"Size: {att.get('size',0)} bytes")
                if att.get("parsedContent"):
                    lines.append("Parsed Content:")
                    lines.append(str(att.get("parsedContent")))
                if att.get("projectInfo"):
                    lines.append("Project Information:")
                    for k, v in att.get("projectInfo", {}).items():
                        if v and v != "unknown":
                            lines.append(f"  {k}: {v}")
                lines.append("-" * 30)
        return "\n".join(lines)

    def _emails_as_text(self, emails: List[Dict]) -> str:
        parts = []
        for i, e in enumerate(emails, 1):
            parts.append(f"Email {i}:\nSubject: {e.get('subject','')}\n"
                         f"From: {e.get('from',{})}\n"
                         f"Content: {e.get('bodyPreview', e.get('body',''))}\n"
                         f"{'-'*20}")
        return "\n".join(parts)


# Global instance (swap your old one)
ai_summary_service = AISummaryServiceLC(
    #model_name=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
    #temperature=float(os.getenv("OPENAI_TEMPERATURE", "0.2"))
)
