"""
AIML Hackathon Problem Statement ID: 26100
Organization: Chennai Petroleum Corporation Limited (CPCL)
Theme: Smart Automation

Test Suite: AI Document Verification Module
File: tests/test_document_verification.py

Verifies:
1. PDF text extraction
2. TXT text extraction
3. Keyword detection
4. Missing keyword detection
5. Document completeness calculation
6. Consistency checking
7. Empty input handling
8. Unsupported file handling
"""

import io
import pytest
from pathlib import Path

from src.document_verification import (
    extract_text_from_pdf,
    extract_text_from_txt,
    extract_text_from_file,
    identify_document_type,
    verify_document_content,
    calculate_document_completeness,
    calculate_document_consistency,
    generate_document_findings,
    SUPPORTED_DOCUMENT_TYPES,
    EXPECTED_KEYWORDS,
    STATUS_VERIFIED,
    STATUS_WARNING,
    STATUS_EXTRACTION_FAILED,
    STATUS_UNSUPPORTED
)

# Minimal valid synthetic PDF containing synthetic test strings
SYNTHETIC_PDF_BYTES = (
    b"%PDF-1.4\n"
    b"1 0 obj << /Type /Catalog /Pages 2 0 R >> endobj\n"
    b"2 0 obj << /Type /Pages /Kids [3 0 R] /Count 1 >> endobj\n"
    b"3 0 obj << /Type /Page /Parent 2 0 R /MediaBox [0 0 300 144] /Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >> endobj\n"
    b"4 0 obj << /Length 72 >> stream\n"
    b"BT\n"
    b"/F1 12 Tf\n"
    b"10 70 Td\n"
    b"(GST GSTIN Goods and Services Tax M/s Bharat Data Systems) Tj\n"
    b"ET\n"
    b"endstream endobj\n"
    b"5 0 obj << /Type /Font /Subtype /Type1 /BaseFont /Helvetica >> endobj\n"
    b"xref\n"
    b"0 6\n"
    b"0000000000 65535 f \n"
    b"0000000009 00000 n \n"
    b"0000000058 00000 n \n"
    b"0000000115 00000 n \n"
    b"0000000244 00000 n \n"
    b"0000000367 00000 n \n"
    b"trailer << /Size 6 /Root 1 0 R >>\n"
    b"startxref\n"
    b"441\n"
    b"%%EOF\n"
)


# -------------------------------------------------------------
# 1. PDF EXTRACTION TEST
# -------------------------------------------------------------
def test_pdf_extraction():
    """Validates that text is safely extracted from a synthetic PDF."""
    pdf_stream = io.BytesIO(SYNTHETIC_PDF_BYTES)
    text, success, error_msg = extract_text_from_pdf(pdf_stream)
    assert success is True
    assert error_msg is None
    assert "GST" in text
    assert "Goods and Services Tax" in text

    # Universal extractor
    res = extract_text_from_file(io.BytesIO(SYNTHETIC_PDF_BYTES), "sample_cert.pdf")
    assert res["status"] == "SUCCESS"
    assert "GSTIN" in res["text"]
    assert res["is_empty"] is False


# -------------------------------------------------------------
# 2. TXT EXTRACTION TEST
# -------------------------------------------------------------
def test_txt_extraction():
    """Validates that plain text documents are correctly extracted."""
    sample_text = (
        "GOVERNMENT OF INDIA\n"
        "PERMANENT ACCOUNT NUMBER CARD\n"
        "Name: Bharat Data Systems Pvt Ltd\n"
        "PAN: ABCDE1234F\n"
        "Permanent Account Number verification document."
    )
    txt_bytes = sample_text.encode("utf-8")
    text, success, error_msg = extract_text_from_txt(io.BytesIO(txt_bytes))
    assert success is True
    assert error_msg is None
    assert "ABCDE1234F" in text
    assert "Permanent Account Number" in text

    res = extract_text_from_file(io.BytesIO(txt_bytes), "pan_card.txt")
    assert res["status"] == "SUCCESS"
    assert "Permanent Account Number" in res["text"]


# -------------------------------------------------------------
# 3. KEYWORD DETECTION TEST
# -------------------------------------------------------------
def test_keyword_detection():
    """Validates that expected keywords for each document type trigger VERIFIED CONTENT."""
    test_cases = [
        (
            "GST Registration Certificate",
            "This document confirms GST registration under GSTIN 27ABCDE1234F1Z5 for Goods and Services Tax compliance."
        ),
        (
            "PAN Card / PAN Document",
            "Income Tax Department PAN card bearing Permanent Account Number ABCDE1234F for bidder."
        ),
        (
            "Udyam Registration Certificate",
            "Ministry of MSME Udyam Registration Number UDYAM-MH-01-0012345 verified."
        ),
        (
            "OEM Authorization Letter",
            "We the Original Equipment Manufacturer hereby issue this OEM Authorization certificate."
        ),
        (
            "EPFO Compliance Document",
            "Employees' Provident Fund EPFO statutory contribution confirmation statement."
        ),
        (
            "ESIC Compliance Document",
            "Employees' State Insurance Corporation ESIC compliance certificate."
        ),
        (
            "Tender Eligibility Document",
            "Procurement tender submission. The Bid meets all eligibility criteria."
        ),
        (
            "Company Registration Document",
            "Ministry of Corporate Affairs Company Registration certificate."
        ),
        (
            "Income Tax Compliance Document",
            "Income Tax Department Assessment Year return and tax clearance certificate."
        )
    ]

    for doc_type, text in test_cases:
        eval_result = verify_document_content(doc_type, text)
        assert eval_result["status"] == STATUS_VERIFIED, f"Failed for {doc_type}"
        assert len(eval_result["missing_keywords"]) == 0
        assert len(eval_result["detected_keywords"]) == len(EXPECTED_KEYWORDS[doc_type])


# -------------------------------------------------------------
# 4. MISSING KEYWORD DETECTION TEST
# -------------------------------------------------------------
def test_missing_keyword_detection():
    """Validates that missing required terms trigger CONTENT WARNING and record missing keywords."""
    # GST missing 'GSTIN' and 'Goods and Services Tax'
    partial_gst_text = "This is a tax receipt with GST mention only."
    res_gst = verify_document_content("GST Registration Certificate", partial_gst_text)
    assert res_gst["status"] == STATUS_WARNING
    assert "GST" in res_gst["detected_keywords"]
    assert "GSTIN" in res_gst["missing_keywords"]
    assert "Goods and Services Tax" in res_gst["missing_keywords"]

    # OEM missing 'Original Equipment Manufacturer'
    partial_oem_text = "OEM Authorization letter for vendor."
    res_oem = verify_document_content("OEM Authorization Letter", partial_oem_text)
    assert res_oem["status"] == STATUS_WARNING
    assert "OEM" in res_oem["detected_keywords"]
    assert "Authorization" in res_oem["detected_keywords"]
    assert "Original Equipment Manufacturer" in res_oem["missing_keywords"]


# -------------------------------------------------------------
# 5. DOCUMENT COMPLETENESS CALCULATION TEST
# -------------------------------------------------------------
def test_document_completeness_calculation():
    """Validates completeness percentage and submission tracking."""
    required = [
        "GST Registration Certificate",
        "PAN Card / PAN Document",
        "Udyam Registration Certificate",
        "Income Tax Compliance Document",
        "OEM Authorization Letter"
    ]

    # Case A: 4 of 5 submitted (80.0%)
    submitted_4 = [
        "GST Registration Certificate",
        "PAN Card / PAN Document",
        "Udyam Registration Certificate",
        "Income Tax Compliance Document"
    ]
    res_4 = calculate_document_completeness(submitted_4, required)
    assert res_4["completeness_percentage"] == 80.0
    assert res_4["required_count"] == 5
    assert res_4["submitted_count"] == 4
    assert res_4["missing_required_docs"] == ["OEM Authorization Letter"]

    # Case B: All 5 submitted (100.0%)
    res_5 = calculate_document_completeness(required, required)
    assert res_5["completeness_percentage"] == 100.0
    assert len(res_5["missing_required_docs"]) == 0

    # Case C: 0 submitted (0.0%)
    res_0 = calculate_document_completeness([], required)
    assert res_0["completeness_percentage"] == 0.0
    assert len(res_0["missing_required_docs"]) == 5

    # Case D: Extra non-required submitted (does not exceed 100%)
    submitted_extra = required + ["Other Procurement Document"]
    res_extra = calculate_document_completeness(submitted_extra, required)
    assert res_extra["completeness_percentage"] == 100.0


# -------------------------------------------------------------
# 6. CONSISTENCY CHECKING TEST
# -------------------------------------------------------------
def test_consistency_checking():
    """Validates heuristic entity and identifier consistency checks."""
    # Scenario A: Consistent company names
    docs_consistent = [
        {
            "document_type": "GST Registration Certificate",
            "file_name": "gst.pdf",
            "text": "Legal Name: Bharat Data Systems\nGSTIN: 27ABCDE1234F1Z5 Goods and Services Tax"
        },
        {
            "document_type": "PAN Card / PAN Document",
            "file_name": "pan.txt",
            "text": "Permanent Account Number\nName: Bharat Data Systems\nPAN: ABCDE1234F"
        }
    ]
    res_cons = calculate_document_consistency(docs_consistent)
    assert res_cons["status"] == "CONSISTENT"
    assert res_cons["score"] >= 90.0

    # Scenario B: Slight name variation (Pvt Ltd)
    docs_variation = [
        {
            "document_type": "GST Registration Certificate",
            "file_name": "gst.pdf",
            "text": "Legal Name: Bharat Data Systems\nGSTIN: 27ABCDE1234F1Z5"
        },
        {
            "document_type": "Company Registration Document",
            "file_name": "company.txt",
            "text": "Company Registration\nCompany Name: Bharat Data Systems Pvt Ltd"
        }
    ]
    res_var = calculate_document_consistency(docs_variation)
    assert res_var["status"] == "CONSISTENT"
    assert any("variation" in w.lower() for w in res_var["warnings"])

    # Scenario C: Inconsistent names (Different bidders)
    docs_mismatch = [
        {
            "document_type": "GST Registration Certificate",
            "file_name": "gst.pdf",
            "text": "Legal Name: Bharat Data Systems\nGSTIN: 27ABCDE1234F1Z5"
        },
        {
            "document_type": "OEM Authorization Letter",
            "file_name": "oem.pdf",
            "text": "OEM Authorization\nIssued to: Zenith Global Logistics Corp\nAuthorization"
        }
    ]
    res_mis = calculate_document_consistency(docs_mismatch)
    assert res_mis["status"] == "REVIEW REQUIRED"
    assert res_mis["score"] <= 65.0
    assert any("different bidder name" in w.lower() for w in res_mis["warnings"])

    # Scenario D: Inconsistent PAN numbers
    docs_pan_mismatch = [
        {
            "document_type": "PAN Card / PAN Document",
            "file_name": "pan1.txt",
            "text": "Name: Bharat Data Systems\nPAN: ABCDE1234F"
        },
        {
            "document_type": "Income Tax Compliance Document",
            "file_name": "tax.txt",
            "text": "Name: Bharat Data Systems\nPAN: ZZZZZ9999Z\nIncome Tax Assessment"
        }
    ]
    res_pan_mis = calculate_document_consistency(docs_pan_mismatch)
    assert res_pan_mis["status"] == "REVIEW REQUIRED"
    assert any("pan numbers" in w.lower() for w in res_pan_mis["warnings"])


# -------------------------------------------------------------
# 7. EMPTY INPUT HANDLING TEST
# -------------------------------------------------------------
def test_empty_input_handling():
    """Validates that empty files and strings do not cause unhandled crashes."""
    # Empty TXT
    res_empty_txt = extract_text_from_file(io.BytesIO(b"   \n\t  "), "blank.txt")
    assert res_empty_txt["status"] == STATUS_WARNING
    assert res_empty_txt["is_empty"] is True

    # Empty content verification
    res_eval = verify_document_content("GST Registration Certificate", "")
    assert res_eval["status"] == STATUS_WARNING
    assert "No readable text was detected" in res_eval["warnings"][0]

    # None / empty consistency
    res_empty_cons = calculate_document_consistency([])
    assert res_empty_cons["status"] == "CONSISTENT"


# -------------------------------------------------------------
# 8. UNSUPPORTED FILE HANDLING TEST
# -------------------------------------------------------------
def test_unsupported_file_handling():
    """Validates graceful rejection of unsupported document extensions."""
    res_docx = extract_text_from_file(io.BytesIO(b"fake docx content"), "tender_bid.docx")
    assert res_docx["status"] == STATUS_UNSUPPORTED
    assert "Unsupported format '.docx'" in res_docx["error"]

    res_jpg = extract_text_from_file(io.BytesIO(b"fake image content"), "certificate.jpg")
    assert res_jpg["status"] == STATUS_UNSUPPORTED
    assert "Unsupported format '.jpg'" in res_jpg["error"]

    findings = generate_document_findings("bid_spec.xlsx", "Other Procurement Document", res_docx)
    assert findings["status"] == STATUS_UNSUPPORTED
    assert findings["status_badge"] == "✗ Failed"


# -------------------------------------------------------------
# 9. CORRUPTED PDF HANDLING TEST
# -------------------------------------------------------------
def test_corrupted_pdf_handling():
    """Validates that corrupted PDF streams are intercepted cleanly."""
    corrupted_stream = io.BytesIO(b"THIS IS NOT A VALID PDF FILE")
    text, success, error_msg = extract_text_from_pdf(corrupted_stream)
    assert success is False
    assert error_msg is not None

    res = extract_text_from_file(corrupted_stream, "corrupted.pdf")
    assert res["status"] == STATUS_EXTRACTION_FAILED
    assert "Text extraction could not be completed" in res["error"]


# -------------------------------------------------------------
# 10. DOCUMENT TYPE IDENTIFICATION HELPER
# -------------------------------------------------------------
def test_document_type_auto_identification():
    """Validates document type heuristic keyword inference."""
    sample = "Employees' Provident Fund EPFO monthly return submission statement."
    identified = identify_document_type(sample)
    assert identified["identified_type"] == "EPFO Compliance Document"
    assert "EPFO" in identified["matched_keywords"]
