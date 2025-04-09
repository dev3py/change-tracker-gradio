from PyPDF2 import PdfReader, PdfWriter

pdf_path = "D:/NIPL-PROJECTS/change-tracker-gradio/TCP-Z1503PSR-TRF1101 (002).pdf"
pdf_save_path = "D:/NIPL-PROJECTS/change-tracker-gradio/output.pdf"

reader = PdfReader(pdf_path)
writer = PdfWriter()

for page in reader.pages:
    # Remove annotations by deleting the "/Annots" entry if it exists
    if "/Annots" in page:
        del page["/Annots"]
    writer.add_page(page)

with open(pdf_save_path, "wb") as f:
    writer.write(f)
