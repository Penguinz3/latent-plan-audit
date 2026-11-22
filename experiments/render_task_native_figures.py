"""Vector publication figures; no scientific computation or simulator imports."""
from pathlib import Path
import argparse

import fitz
from reportlab.pdfgen import canvas
from reportlab.lib.colors import HexColor


def overview(path):
    c = canvas.Canvas(str(path), pagesize=(600, 280))
    c.setTitle("Fixed-candidate decision audit")
    blue, teal, orange = "#2457A7", "#17756D", "#9A5A16"

    def text(x, y, lines, size=12, bold=False):
        c.setFillColor(HexColor("#17232E"))
        c.setFont("Helvetica-Bold" if bold else "Helvetica", size)
        for line in lines:
            c.drawString(x, y, line)
            y -= size + 3

    def box(x, y, w, h, lines, colour):
        c.setStrokeColor(HexColor(colour))
        c.setFillColor(HexColor("#FAFBFC"))
        c.roundRect(x, y, w, h, 4, fill=1, stroke=1)
        text(x + 7, y + h - 17, lines)

    def arrow(points, colour):
        c.setStrokeColor(HexColor(colour)); c.setFillColor(HexColor(colour))
        c.setLineWidth(1.25)
        p = c.beginPath(); p.moveTo(*points[0])
        for point in points[1:]:
            p.lineTo(*point)
        c.drawPath(p)
        x, y = points[-1]
        p = c.beginPath(); p.moveTo(x, y); p.lineTo(x-5, y+3); p.lineTo(x-5, y-3); p.close()
        c.drawPath(p, fill=1, stroke=0)

    text(8, 264, ["Same 300 candidates, start and goal. No CEM updates or closed-loop execution."], 12, True)
    box(8, 126, 115, 66, ["Fixed candidate set", "300 complete", "action sequences"], "#617383")
    lanes = [(203, blue, ["World-model", "terminal prediction"], ["Predicted latent", "goal cost"], ["Choose", "a_pred"]),
             (129, teal, ["Simulator replay", "+ same encoder"], ["Realized latent", "goal cost"], ["Choose", "a_real"]),
             (55, orange, ["Same simulator", "replay evidence"], ["Native terminal", "state distance"], ["Choose", "a_state"])]
    for y, colour, source, cost, selection in lanes:
        box(143, y, 123, 49, source, colour)
        box(282, y, 118, 49, cost, colour)
        box(416, y, 82, 49, selection, colour)
        arrow([(123, 159), (133, 159), (133, y+24.5), (143, y+24.5)], colour)
        arrow([(266, y+24.5), (282, y+24.5)], colour)
        arrow([(400, y+24.5), (416, y+24.5)], colour)
        arrow([(498, y+24.5), (514, y+24.5)], colour)
    box(514, 55, 78, 197, ["Evaluate", "selected", "actions:", "", "distance", "utility", "and task", "outcome"], "#617383")
    text(8, 33, ["Predicted -> realized: only the terminal latent input changes.",
                 "State arm: bypasses encoder and latent objective; retrospective, not deployable."], 12)
    c.showPage(); c.save()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--ladder-svg", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    overview(args.output_dir / "audit_overview.pdf")
    with fitz.open(args.ladder_svg) as source:
        vector_pdf = source.convert_to_pdf()
    with fitz.open("pdf", vector_pdf) as converted:
        converted.save(args.output_dir / "intervention_ladder.pdf")
    print("Created two vector figures; scientific summaries unchanged.")


if __name__ == "__main__":
    main()
