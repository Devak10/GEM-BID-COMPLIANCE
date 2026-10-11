"""
AI-Powered Integrated Bid Compliance Verification Platform for GeM Procurement
AIML Hackathon Problem Statement ID: 26100
Organization: Chennai Petroleum Corporation Limited (CPCL)
Theme: Smart Automation

Module: Document Verification Engine
File: src/document_verification.py

Provides reusable functions for prototype procurement document verification:
- PDF and TXT text extraction
- Document type identification & keyword checking
- Content verification status determination
- Required document completeness calculation
- Heuristic cross-document consistency checks
- Document findings and warnings generation
"""

import io
import re
from pathlib import Path
from difflib import SequenceMatcher
from typing import Dict, Any, List, Optional, Tuple, Union

try:
    from pypdf import PdfReader
    PYPDF_AVAILABLE = True
except ImportError:
    PYPDF_AVAILABLE = False


# Supported Prototype Document Types
SUPPORTED_DOCUMENT_TYPES = [
    "GST Registration Certificate",
    "PAN Card / PAN Document",
    "Udyam Registration Certificate",
    "Income Tax Compliance Document",
    "OEM Authorization Letter",
    "EPFO Compliance Document",
    "ESIC Compliance Document",
    "Tender Eligibility Document",
    "Company Registration Document",
    "Other Procurement Document"
]

# Prototype Expected Keywords Mapping
EXPECTED_KEYWORDS: Dict[str, List[str]] = {
    "GST Registration Certificate": [
        "GST",
        "GSTIN",
        "Goods and Services Tax"
    ],
    "PAN Card / PAN Document": [
        "PAN",
        "Permanent Account Number"
    ],
    "Udyam Registration Certificate": [
        "Udyam",
        "MSME",
        "Registration Number"
    ],
    "OEM Authorization Letter": [
        "OEM",
        "Original Equipment Manufacturer",
        "Authorization"
    ],
    "EPFO Compliance Document": [
        "EPFO",
        "Employees' Provident Fund"
    ],
    "ESIC Compliance Document": [
        "ESIC",
        "Employees' State Insurance"
    ],
    "Tender Eligibility Document": [
        "Tender",
        "Eligibility",
        "Bid",
        "Procurement"
    ],
    "Company Registration Document": [
        "Company",
        "Registration",
        "Corporate"
    ],
    "Income Tax Compliance Document": [
        "Income Tax",
        "Tax",
        "Assessment"
    ],
    "Other Procurement Document": []
}

# Status Constants
STATUS_VERIFIED = "VERIFIED CONTENT"
STATUS_WARNING = "CONTENT WARNING"
STATUS_EXTRACTION_FAILED = "TEXT EXTRACTION FAILED"
STATUS_UNSUPPORTED = "UNSUPPORTED FORMAT"


def extract_text_from_pdf(file_source: Union[str, Path, io.BytesIO, bytes, Any]) -> Tuple[str, bool, Optional[str]]:
    """
    Extracts readable text from a PDF file using pypdf.
    
    Args:
        file_source: File path, BytesIO, bytes, or Streamlit UploadedFile.
        
    Returns:
        Tuple of (extracted_text, success_boolean, error_message_if_any)
    """
    if not PYPDF_AVAILABLE:
        return "", False, "pypdf library is not installed."

    try:
        # Normalize file source to stream or path
        stream = None
        if isinstance(file_source, (str, Path)):
            stream = open(file_source, "rb")
        elif isinstance(file_source, bytes):
            stream = io.BytesIO(file_source)
        elif hasattr(file_source, "read"):
            # File-like object (e.g. UploadedFile)
            if hasattr(file_source, "seek"):
                file_source.seek(0)
            data_bytes = file_source.read()
            stream = io.BytesIO(data_bytes)
            if hasattr(file_source, "seek"):
                file_source.seek(0)
        else:
            return "", False, "Invalid file source type provided."

        reader = PdfReader(stream)
        text_parts: List[str] = []
        for page_idx, page in enumerate(reader.pages):
            try:
                page_text = page.extract_text()
                if page_text:
                    text_parts.append(page_text)
            except Exception as page_err:
                text_parts.append(f"[Page {page_idx + 1} extraction notice: {page_err}]")

        full_text = "\n".join(text_parts).strip()
        return full_text, True, None

    except Exception as e:
        return "", False, str(e)


def extract_text_from_txt(file_source: Union[str, Path, io.BytesIO, bytes, Any]) -> Tuple[str, bool, Optional[str]]:
    """
    Extracts text from a plain text file.
    
    Args:
        file_source: File path, BytesIO, bytes, or Streamlit UploadedFile.
        
    Returns:
        Tuple of (extracted_text, success_boolean, error_message_if_any)
    """
    try:
        raw_bytes = b""
        if isinstance(file_source, (str, Path)):
            with open(file_source, "rb") as f:
                raw_bytes = f.read()
        elif isinstance(file_source, bytes):
            raw_bytes = file_source
        elif hasattr(file_source, "read"):
            if hasattr(file_source, "seek"):
                file_source.seek(0)
            raw_bytes = file_source.read()
            if hasattr(file_source, "seek"):
                file_source.seek(0)
        else:
            return "", False, "Invalid file source type provided."

        # Decode with common encodings
        for encoding in ["utf-8", "utf-8-sig", "latin-1", "cp1252"]:
            try:
                decoded_text = raw_bytes.decode(encoding)
                return decoded_text.strip(), True, None
            except UnicodeDecodeError:
                continue

        return "", False, "Unable to decode text file with standard encodings."

    except Exception as e:
        return "", False, str(e)


def extract_text_from_file(file_source: Any, filename: str) -> Dict[str, Any]:
    """
    Universal extraction handler supporting PDF and TXT with friendly error management.
    Never crashes on invalid, corrupted, or unsupported files.
    """
    ext = Path(filename).suffix.lower()

    if ext == ".pdf":
        text, success, error_msg = extract_text_from_pdf(file_source)
        if not success:
            err = "Text extraction could not be completed for this document."
            if error_msg:
                err += f" ({error_msg})"
            return {
                "status": STATUS_EXTRACTION_FAILED,
                "text": "",
                "error": err,
                "technical_error": error_msg,
                "preview": ""
            }
    elif ext == ".txt":
        text, success, error_msg = extract_text_from_txt(file_source)
        if not success:
            err = "Text extraction could not be completed for this document."
            if error_msg:
                err += f" ({error_msg})"
            return {
                "status": STATUS_EXTRACTION_FAILED,
                "text": "",
                "error": err,
                "technical_error": error_msg,
                "preview": ""
            }
    else:
        return {
            "status": STATUS_UNSUPPORTED,
            "text": "",
            "error": f"Unsupported format '{ext}'. Only PDF and TXT documents are supported.",
            "preview": ""
        }

    # Check for empty text
    if not text.strip():
        return {
            "status": STATUS_WARNING,
            "text": "",
            "warning": "Warning: No readable text was detected.",
            "preview": "",
            "is_empty": True
        }

    preview = text[:3000]
    return {
        "status": "SUCCESS",
        "text": text,
        "preview": preview,
        "is_empty": False
    }


def identify_document_type(text: str) -> Dict[str, Any]:
    """
    Analyzes document text against prototype keyword sets to suggest likely document type.
    """
    if not text:
        return {
            "identified_type": "Other Procurement Document",
            "confidence": 0.0,
            "matched_keywords": []
        }

    normalized_text = text.lower()
    scores: Dict[str, List[str]] = {}

    for doc_type, keywords in EXPECTED_KEYWORDS.items():
        if not keywords:
            continue
        matches = [kw for kw in keywords if kw.lower() in normalized_text]
        scores[doc_type] = matches

    best_match = "Other Procurement Document"
    best_count = 0
    best_keywords: List[str] = []

    for doc_type, matched in scores.items():
        if len(matched) > best_count:
            best_count = len(matched)
            best_match = doc_type
            best_keywords = matched

    expected_total = len(EXPECTED_KEYWORDS.get(best_match, []))
    confidence = (best_count / expected_total) if expected_total > 0 else 0.0

    return {
        "identified_type": best_match,
        "confidence": round(confidence, 2),
        "matched_keywords": best_keywords
    }


def verify_document_content(doc_type: str, text: str) -> Dict[str, Any]:
    """
    Performs prototype keyword-based content verification for a selected document type.
    
    Status results:
    - VERIFIED CONTENT: All expected keywords detected.
    - CONTENT WARNING: Missing expected keywords or empty text.
    - Other Procurement Document: General extraction without official validation.
    """
    if not text or not text.strip():
        return {
            "doc_type": doc_type,
            "status": STATUS_WARNING,
            "detected_keywords": [],
            "missing_keywords": EXPECTED_KEYWORDS.get(doc_type, []),
            "warnings": ["Warning: No readable text was detected."],
            "findings": "Document appears empty or contains no extractable text."
        }

    # Special handling for Other Procurement Document
    if doc_type == "Other Procurement Document":
        return {
            "doc_type": doc_type,
            "status": STATUS_WARNING,
            "detected_keywords": [],
            "missing_keywords": [],
            "warnings": ["Prototype content verification — not official document validation."],
            "findings": "General procurement document: General text extraction completed. Document not evaluated under standardized statutory rules."
        }

    expected = EXPECTED_KEYWORDS.get(doc_type, [])
    if not expected:
        return {
            "doc_type": doc_type,
            "status": STATUS_WARNING,
            "detected_keywords": [],
            "missing_keywords": [],
            "warnings": ["Unrecognized document category."],
            "findings": "No verification criteria defined for this document category."
        }

    normalized_text = text.lower()
    detected: List[str] = []
    missing: List[str] = []

    for kw in expected:
        # Check case-insensitive keyword or phrase presence
        if kw.lower() in normalized_text:
            detected.append(kw)
        else:
            missing.append(kw)

    warnings: List[str] = []
    if missing:
        status = STATUS_WARNING
        warnings.append(f"Expected {doc_type} terminology was not detected.")
        findings = f"Missing expected keywords: {', '.join(missing)}."
    else:
        status = STATUS_VERIFIED
        findings = f"Expected {doc_type} terminology detected."

    return {
        "doc_type": doc_type,
        "status": status,
        "detected_keywords": detected,
        "missing_keywords": missing,
        "warnings": warnings,
        "findings": findings
    }


def extract_bidder_name_heuristically(text: str) -> Optional[str]:
    """
    Heuristically extracts company/bidder name or key entity identifiers from document text.
    Uses regex patterns common in Indian statutory and commercial tender documents.
    """
    if not text:
        return None

    patterns = [
        # Explicit labels
        r"(?:Legal Name|Trade Name|Company Name|Bidder Name|Name of the Bidder|Name of Enterprise|Vendor Name|Firm Name|Name)\s*[:\-]\s*([A-Za-z0-9\s&.,'-]+?)(?=\n|\r|,|CIN|GSTIN|PAN|$)",
        # M/s Pattern
        r"(?:M/s\.?|M/S\.?|Messrs\.?)\s+([A-Za-z0-9\s&.,'-]+?)(?=\s+(?:Private|Pvt|Limited|Ltd|LLP|Inc|\n|\r|,|CIN|GSTIN|PAN))",
        # Issued to
        r"(?:Issued to|In favour of|Awarded to)\s*[:\-]?\s*([A-Za-z0-9\s&.,'-]+?)(?=\n|\r|,|CIN|GSTIN|PAN|$)",
        # Authorization Partner / Name
        r"(?:authorizes|appoints)\s+(?:M/s\.?\s+)?([A-Za-z0-9\s&.,'-]+?)(?=\s+as|\s+to|\n|\r|,|$)"
    ]

    for pat in patterns:
        match = re.search(pat, text, re.IGNORECASE)
        if match:
            candidate = match.group(1).strip()
            # Clean trailing punctuation
            candidate = re.sub(r"^[^\w]+|[^\w]+$", "", candidate)
            if len(candidate) >= 3 and not candidate.lower().startswith("the "):
                return candidate

    return None


def normalize_company_name(name: str) -> str:
    """Normalizes company name for similarity comparison."""
    if not name:
        return ""
    cleaned = re.sub(r"(?i)\b(private|pvt|limited|ltd|llp|inc|corp|corporation|enterprises|enterprise|solutions|services|company)\b", "", name)
    cleaned = re.sub(r"[^\w\s]", "", cleaned)
    return " ".join(cleaned.lower().split())


def calculate_document_completeness(submitted_doc_types: List[str], required_doc_types: List[str]) -> Dict[str, Any]:
    """
    Calculates completeness percentage of required bid documents.
    
    Formula:
    Document Completeness % = (Number of submitted required document types / Total required document types) * 100
    """
    if not required_doc_types:
        return {
            "completeness_percentage": 100.0,
            "required_count": 0,
            "submitted_count": len(submitted_doc_types),
            "submitted_required_docs": [],
            "missing_required_docs": [],
            "all_submitted_docs": list(set(submitted_doc_types))
        }

    submitted_set = set(submitted_doc_types)
    required_set = set(required_doc_types)

    submitted_required = list(submitted_set.intersection(required_set))
    missing_required = list(required_set.difference(submitted_set))

    completeness = round((len(submitted_required) / len(required_set)) * 100.0, 1)

    return {
        "completeness_percentage": completeness,
        "required_count": len(required_set),
        "submitted_count": len(submitted_required),
        "submitted_required_docs": sorted(submitted_required),
        "missing_required_docs": sorted(missing_required),
        "all_submitted_docs": sorted(list(submitted_set))
    }


def calculate_document_consistency(documents: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Performs heuristic consistency check across uploaded documents.
    Extracts entity names and checks cross-document alignment.
    
    Returns:
    - consistency_status: "CONSISTENT" or "REVIEW REQUIRED"
    - consistency_score: float (0.0 to 100.0)
    - warnings: List[str]
    - extracted_entities: Dict of extracted names/IDs
    """
    if not documents:
        return {
            "status": "CONSISTENT",
            "score": 100.0,
            "warnings": [],
            "extracted_entities": {},
            "details": "No documents uploaded for consistency comparison."
        }

    extracted_names: Dict[str, str] = {}
    extracted_identifiers: Dict[str, Dict[str, str]] = {}

    for doc in documents:
        doc_type = doc.get("document_type", "Unknown")
        file_name = doc.get("file_name", "Unknown File")
        text = doc.get("text", "")

        name = extract_bidder_name_heuristically(text)
        if name:
            extracted_names[f"{doc_type} ({file_name})"] = name

        # Extract potential statutory IDs (GSTIN, PAN)
        gstin_match = re.search(r"\b[0-9]{2}[A-Z]{5}[0-9]{4}[A-Z]{1}[1-9A-Z]{1}Z[0-9A-Z]{1}\b", text)
        pan_match = re.search(r"\b[A-Z]{5}[0-9]{4}[A-Z]{1}\b", text)
        ids = {}
        if gstin_match:
            ids["GSTIN"] = gstin_match.group(0)
        if pan_match:
            ids["PAN"] = pan_match.group(0)
        if ids:
            extracted_identifiers[f"{doc_type} ({file_name})"] = ids

    warnings: List[str] = []
    names_list = list(extracted_names.values())
    has_major_mismatch = False
    has_slight_variation = False

    # 1. Company Name Comparison
    if len(names_list) >= 2:
        base_name = names_list[0]
        base_norm = normalize_company_name(base_name)

        for other_name in names_list[1:]:
            other_norm = normalize_company_name(other_name)
            ratio = SequenceMatcher(None, base_norm, other_norm).ratio()

            if ratio < 0.60:
                has_major_mismatch = True
                warnings.append(
                    f"Different bidder name across documents: '{base_name}' vs '{other_name}'."
                )
            elif ratio < 0.95 or base_name.lower().strip() != other_name.lower().strip():
                has_slight_variation = True
                warnings.append(
                    f"Possible company-name variation detected: '{base_name}' vs '{other_name}'."
                )

    # 2. Statutory Identifiers Cross-Check (PAN & GSTIN)
    pans_found = {v["PAN"] for v in extracted_identifiers.values() if "PAN" in v}
    if len(pans_found) > 1:
        has_major_mismatch = True
        warnings.append(f"Discrepancy in PAN numbers detected across documents: {', '.join(sorted(pans_found))}.")

    gstins_found = {v["GSTIN"] for v in extracted_identifiers.values() if "GSTIN" in v}
    if len(gstins_found) > 1:
        has_major_mismatch = True
        warnings.append(f"Discrepancy in GSTIN numbers detected across documents: {', '.join(sorted(gstins_found))}.")

    # 3. Overall Consistency Evaluation
    if has_major_mismatch:
        status = "REVIEW REQUIRED"
        score = 60.0
        details = "Meaningful discrepancies detected between document entity names or statutory identifiers."
    elif has_slight_variation:
        status = "CONSISTENT"
        score = 85.0
        details = "Minor company name variations detected across documents; overall consistency satisfactory."
    elif len(names_list) >= 1 or len(pans_found) == 1 or len(gstins_found) == 1:
        status = "CONSISTENT"
        score = 95.0
        details = "High entity consistency confirmed across all uploaded documents."
    else:
        # No names or identifiers extracted
        status = "CONSISTENT"
        score = 80.0
        warnings.append("Missing expected identifier or company name across uploaded documents.")
        details = "Could not automatically extract entity names for automated comparison. Officer review recommended."

    return {
        "status": status,
        "score": score,
        "warnings": warnings,
        "extracted_entities": extracted_names,
        "extracted_identifiers": extracted_identifiers,
        "details": details
    }


def generate_document_findings(
    file_name: str,
    doc_type: str,
    extraction_result: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Assembles comprehensive findings, detected keywords, warnings, and status for UI display.
    """
    status_raw = extraction_result.get("status", STATUS_EXTRACTION_FAILED)
    text = extraction_result.get("text", "")
    preview = extraction_result.get("preview", "")

    if status_raw == STATUS_UNSUPPORTED:
        return {
            "document_type": doc_type,
            "file_name": file_name,
            "status": STATUS_UNSUPPORTED,
            "status_badge": "✗ Failed",
            "detected_keywords": [],
            "missing_keywords": EXPECTED_KEYWORDS.get(doc_type, []),
            "warnings": [extraction_result.get("error", "Unsupported file format.")],
            "extracted_text_preview": "",
            "findings": "File format not supported. Only PDF and TXT files are accepted."
        }

    if status_raw == STATUS_EXTRACTION_FAILED:
        return {
            "document_type": doc_type,
            "file_name": file_name,
            "status": STATUS_EXTRACTION_FAILED,
            "status_badge": "✗ Failed",
            "detected_keywords": [],
            "missing_keywords": EXPECTED_KEYWORDS.get(doc_type, []),
            "warnings": ["Text extraction could not be completed for this document."],
            "extracted_text_preview": "",
            "findings": extraction_result.get("error", "Text extraction failed.")
        }

    # Verify content
    content_eval = verify_document_content(doc_type, text)
    final_status = content_eval["status"]
    badge = "✓ Passed" if final_status == STATUS_VERIFIED else "⚠ Warning"

    return {
        "document_type": doc_type,
        "file_name": file_name,
        "status": final_status,
        "status_badge": badge,
        "detected_keywords": content_eval["detected_keywords"],
        "missing_keywords": content_eval["missing_keywords"],
        "warnings": content_eval["warnings"],
        "extracted_text_preview": preview,
        "findings": content_eval["findings"]
    }
