import fitz  # PyMuPDF
import json
import os

# Define input and output paths
input_folder = "D:\uni\Sem 6\LLMs\legal-compliance-search\data\raw_docs"  # Update with actual path if different
output_file = "D:\uni\Sem 6\LLMs\legal-compliance-search\data\processed_docs\structured_compliance_data.json"

# List of uploaded documents
pdf_files = [
    "GDPR.pdf",
    "ITACT.pdf",
    "PCI-DSS-QUICK_REF.pdf",
    "RBI-Compliance.pdf",
    "RBI-Compliance-FAQ.pdf"
]

def extract_text_from_pdf(pdf_path):
    """Extracts text from a given PDF file."""
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text("text") + "\n"
    return text

def structure_text(text, filename):
    """Splits text into structured format using basic section detection."""
    sections = []
    current_section = {"title": "Introduction", "content": ""}
    
    for line in text.split("\n"):
        line = line.strip()
        if len(line) < 3:
            continue  # Skip short lines

        # Detect section headers (basic heuristic: lines in ALL CAPS or numbered headings)
        if line.isupper() or line.startswith(("CHAPTER", "SECTION", "Article", "Rule", "Guideline")):
            if current_section["content"]:  # Save previous section
                sections.append(current_section)
            current_section = {"title": line, "content": ""}
        else:
            current_section["content"] += line + " "

    if current_section["content"]:  # Save last section
        sections.append(current_section)

    return {"filename": filename, "sections": sections}

# Process all PDFs
structured_data = []

for pdf in pdf_files:
    pdf_path = os.path.join(input_folder, pdf)
    if os.path.exists(pdf_path):
        text = extract_text_from_pdf(pdf_path)
        structured_data.append(structure_text(text, pdf))

# Save structured data as JSON
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(structured_data, f, indent=4)

print(f"Structured compliance data saved to {output_file}")
