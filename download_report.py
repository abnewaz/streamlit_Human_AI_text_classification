"""
PDF Report Generator for the AI vs Human Text Classifier.
Converts the markdown report (stats + prediction) into a downloadable PDF.
"""

import re
from fpdf import FPDF


class ReportPDF(FPDF):
    """Custom PDF class for report generation."""

    def __init__(self):
        super().__init__()
        self.set_auto_page_break(auto=True, margin=20)

    def header(self):
        if self.page_no() == 1:
            self.set_font("Helvetica", "B", 16)
            self.cell(0, 10, "AI vs Human Text Classifier - Report", ln=True, align="C")
            self.ln(4)

    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.cell(0, 10, f"Page {self.page_no()}/{{nb}}", align="C")


def _sanitize_text(text: str) -> str:
    """Replace characters that cannot be encoded in latin-1 with ASCII equivalents."""
    replacements = {
        "\u201c": '"',
        "\u201d": '"',
        "\u2018": "'",
        "\u2019": "'",
        "\u2013": "-",
        "\u2014": "--",
        "\u2026": "...",
        "\u00e9": "e",
        "\u00e8": "e",
        "\u00ea": "e",
        "\u00e0": "a",
        "\u00e2": "a",
        "\u00f9": "u",
        "\u00fb": "u",
        "\u00e7": "c",
        "\u00ef": "i",
        "\u00ee": "i",
        "\u00f4": "o",
        "\u00f6": "oe",
        "\u00fc": "ue",
        "\u00df": "ss",
        "\u00b0": " deg",
    }
    # Replace known Unicode characters
    for src, dst in replacements.items():
        text = text.replace(src, dst)
    # Remove any remaining non-latin-1 characters
    text = text.encode("latin-1", errors="replace").decode("latin-1")
    return text


def _render_markdown_simple(pdf: FPDF, markdown_text: str):
    """
    Render a subset of markdown into the PDF:
    - Lines starting with `#` are headers (bold, larger)
    - Lines starting with `**...**` are bold
    - Table rows starting with `|` are rendered as table cells
    - Blank lines add spacing
    - Normal text is wrapped
    """
    lines = markdown_text.split("\n")
    for raw_line in lines:
        line = raw_line.strip()

        # Skip empty lines or horizontal rules
        if not line or line.startswith("---"):
            pdf.ln(2)
            continue

        # ---- Table rows (| ... | ... |) ----
        if line.startswith("|") and line.endswith("|"):
            # Extract cell values between pipes, stripping markdown bold markers
            cells = [
                cell.strip().replace("**", "").replace("*", "")
                for cell in line.split("|")[1:-1]
            ]

            # If it's a separator row (|---|...|), skip it
            if all(re.fullmatch(r"-+", cell) for cell in cells):
                pdf.ln(1)
                continue

            # Render each cell
            for cell in cells:
                clean_cell = _sanitize_text(cell)
                pdf.set_font("Helvetica", "", 9)
                # cell_width = (pdf.w - 2 * pdf.l_margin) / max(len(cells), 1)
                cell_width = (pdf.w - 2 * pdf.l_margin) / max(len(cells), 1)
                pdf.cell(cell_width, 6, clean_cell, border=1, align="C")
            pdf.ln()
            continue

        # ---- Headers: lines starting with # ----
        if line.startswith("##"):
            text = line.lstrip("#").strip()
            pdf.set_font("Helvetica", "B", 13)
            pdf.cell(0, 8, _sanitize_text(text), ln=True)
            pdf.ln(1)
            continue
        if line.startswith("#"):
            text = line.lstrip("#").strip()
            pdf.set_font("Helvetica", "B", 11)
            pdf.cell(0, 7, _sanitize_text(text), ln=True)
            pdf.ln(1)
            continue

        # ---- Bold lines (entire line is bold) ----
        if line.startswith("**") and line.endswith("**"):
            text = line.replace("**", "").strip()
            # Check if it's a list item
            if text.startswith("- "):
                text = text[2:]
                pdf.set_font("Helvetica", "B", 10)
                pdf.cell(5, 6, chr(8226).encode("latin-1", errors="replace").decode("latin-1"), ln=False)  # bullet
                pdf.set_x(pdf.get_x() + 2)
                pdf.multi_cell(0, 6, _sanitize_text(text))
            else:
                pdf.set_font("Helvetica", "B", 10)
                pdf.multi_cell(0, 6, _sanitize_text(text))
            continue

        # ---- Inline bold markers ----
        # Render bold text segments in bold font
        parts = re.split(r"(\*\*.+?\*\*)", line)
        pdf.set_font("Helvetica", "", 10)
        for part in parts:
            if part.startswith("**") and part.endswith("**"):
                bold_text = part[2:-2]
                pdf.set_font("Helvetica", "B", 10)
                pdf.write(6, _sanitize_text(bold_text))
                pdf.set_font("Helvetica", "", 10)
            else:
                pdf.write(6, _sanitize_text(part))
        pdf.ln()
        continue


def generate_report_pdf(stats_markdown: str, response_markdown: str) -> bytes:
    """
    Generate a PDF report from the statistics and prediction response markdown.

    Parameters
    ----------
    stats_markdown : str
        The text statistics markdown.
    response_markdown : str
        The prediction response markdown.

    Returns
    -------
    bytes
        The PDF content as raw bytes, ready for download.
    """
    pdf = ReportPDF()
    pdf.alias_nb_pages()
    pdf.add_page()

    # ---- Section 1: Text Statistics ----
    pdf.set_font("Helvetica", "B", 14)
    pdf.cell(0, 10, "Text Statistics", ln=True)
    pdf.set_draw_color(0, 102, 204)
    pdf.line(pdf.l_margin, pdf.get_y(), pdf.w - pdf.r_margin, pdf.get_y())
    pdf.ln(3)
    _render_markdown_simple(pdf, stats_markdown)

    # ---- Section 2: Prediction Response ----
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 14)
    pdf.cell(0, 10, "Prediction Result", ln=True)
    pdf.set_draw_color(0, 102, 204)
    pdf.line(pdf.l_margin, pdf.get_y(), pdf.w - pdf.r_margin, pdf.get_y())
    pdf.ln(3)
    _render_markdown_simple(pdf, response_markdown)

    # Return the PDF as bytes
    return pdf.output(dest="S").encode("latin-1")