import pdfplumber

pdf_path = "2504.19874.pdf"
output_path = "paper_content.txt"

with pdfplumber.open(pdf_path) as pdf:
    full_text = ""
    for page_num, page in enumerate(pdf.pages, 1):
        text = page.extract_text()
        if text:
            full_text += f"\n--- Page {page_num} ---\n\n"
            full_text += text

with open(output_path, "w", encoding="utf-8") as f:
    f.write(full_text)

print(f"PDF text extracted to {output_path}")
