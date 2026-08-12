"""Create a readable two-page PDF archive from the verified job-page capture."""

from pathlib import Path

from PIL import Image
from reportlab.lib.pagesizes import A4
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = PROJECT_ROOT / "report" / "overleaf"
SOURCE_IMAGE = REPORT_DIR / "job_ad_blackwoods_clean.jpg"
OUTPUT_PDF = REPORT_DIR / "job_ad_blackwoods.pdf"
SOURCE_URL = (
    "https://www.livehire.com/careers/blackwoods/job/"
    "DB9HM/1W7S2DP3U/junior-data-scientist"
)


def main() -> None:
    source = Image.open(SOURCE_IMAGE).convert("RGB")

    # The right column contains application controls rather than the job
    # description. Cropping it increases legibility while preserving the
    # employer's original title, duties, skills, culture and closing details.
    crops = [
        source.crop((45, 65, 885, 1120)),
        source.crop((45, 1000, 885, 1785)),
    ]

    page_width, page_height = A4
    pdf = canvas.Canvas(str(OUTPUT_PDF), pagesize=A4)
    pdf.setTitle("Blackwoods Junior Data Scientist job advertisement")
    pdf.setAuthor("Blackwoods - archived from the public Humanforce Talent listing")
    pdf.setSubject("Appendix copy of the selected job advertisement")

    margin_x = 42
    top_margin = 42
    footer_height = 34
    available_width = page_width - 2 * margin_x
    available_height = page_height - top_margin - footer_height - 18

    for page_number, crop in enumerate(crops, start=1):
        width, height = crop.size
        scale = min(available_width / width, available_height / height)
        draw_width = width * scale
        draw_height = height * scale
        x = (page_width - draw_width) / 2
        y = page_height - top_margin - draw_height

        pdf.drawImage(
            ImageReader(crop),
            x,
            y,
            width=draw_width,
            height=draw_height,
            preserveAspectRatio=True,
            mask="auto",
        )

        pdf.setFont("Helvetica", 7.5)
        footer = (
            "Original public advertisement captured 10 August 2026. "
            f"Page {page_number} of {len(crops)}."
        )
        pdf.drawString(margin_x, 25, footer)
        pdf.setFillColorRGB(0.02, 0.32, 0.50)
        pdf.drawString(
            margin_x,
            14,
            "Clickable original listing: livehire.com/blackwoods/junior-data-scientist",
        )
        pdf.linkURL(
            SOURCE_URL,
            (margin_x, 10, page_width - margin_x, 23),
            relative=0,
        )
        pdf.setFillColorRGB(0, 0, 0)
        pdf.showPage()

    pdf.save()
    print(OUTPUT_PDF)


if __name__ == "__main__":
    main()
