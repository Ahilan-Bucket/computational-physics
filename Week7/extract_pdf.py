import fitz

pdf_path = r"D:\Documents\SFU\Phys395-ComputationalPhysics\computational-physics\Week7\Guide7.pdf"
txt_path = r"D:\Documents\SFU\Phys395-ComputationalPhysics\computational-physics\Week7\Guide7_extracted.txt"

with fitz.open(pdf_path) as doc:
    with open(txt_path, "w", encoding="utf-8") as f:
        for i in range(doc.page_count):
            f.write(f"=== Page {i+1} ===\n")
            f.write(doc[i].get_text())
            f.write("\n\n")

print("Extraction complete.")
