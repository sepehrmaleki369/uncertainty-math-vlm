"""Build the supervisor deck as a .pptx for Google Slides.

Text is deliberately sparse: a slide is a prompt for the speaker, not a
document. Every number here is the frozen value from reference/*.json.

Fonts are restricted to Georgia (headlines) and Arial (everything else)
because Google Slides has both natively -- anything else silently
substitutes on upload and wrecks the layout.
"""
from PIL import Image
from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR

C = {
    "ink":   RGBColor(0x17, 0x1A, 0x21),
    "soft":  RGBColor(0x4A, 0x52, 0x60),
    "faint": RGBColor(0x79, 0x82, 0x8F),
    "rule":  RGBColor(0xDD, 0xE2, 0xEA),
    "accent":RGBColor(0x2F, 0x5D, 0x8C),
    "good":  RGBColor(0x1B, 0x7A, 0x4B),
    "bad":   RGBColor(0xB2, 0x3A, 0x34),
    "warn":  RGBColor(0x9A, 0x6A, 0x12),
    "white": RGBColor(0xFF, 0xFF, 0xFF),
    "panel": RGBColor(0xF5, 0xF7, 0xFA),
}
SERIF, SANS = "Georgia", "Arial"
W, H = Inches(13.333), Inches(7.5)
M = Inches(0.62)                      # page margin
CASES = "reference/cases"

prs = Presentation()
prs.slide_width, prs.slide_height = W, H
BLANK = prs.slide_layouts[6]


def tb(slide, x, y, w, h, align=PP_ALIGN.LEFT, anchor=MSO_ANCHOR.TOP):
    box = slide.shapes.add_textbox(x, y, w, h)
    tf = box.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    tf.vertical_anchor = anchor
    tf.paragraphs[0].alignment = align
    return tf


def run(p, text, size, color, font=SANS, bold=False, italic=False, spacing=None):
    r = p.add_run(); r.text = text
    r.font.size, r.font.name, r.font.bold, r.font.italic = Pt(size), font, bold, italic
    r.font.color.rgb = color
    return r


def eyebrow(slide, text, color):
    tf = tb(slide, M, Inches(0.42), Inches(9), Inches(0.3))
    run(tf.paragraphs[0], text.upper(), 11, color, bold=True)


def title(slide, text, y=Inches(0.78), size=31, width=11.4):
    tf = tb(slide, M, y, Inches(width), Inches(1.0))
    run(tf.paragraphs[0], text, size, C["ink"], font=SERIF, bold=True)


def slide_num(slide, n, total=6):
    tf = tb(slide, W - M - Inches(1.2), H - Inches(0.52), Inches(1.2), Inches(0.3),
            align=PP_ALIGN.RIGHT)
    run(tf.paragraphs[0], f"{n} / {total}", 10, C["faint"])


def rule(slide, y, x=M, w=None):
    w = w or (W - 2 * M)
    ln = slide.shapes.add_shape(1, x, y, w, Emu(9525))   # 1 = rectangle
    ln.fill.solid(); ln.fill.fore_color.rgb = C["rule"]
    ln.line.fill.background(); ln.shadow.inherit = False
    return ln


def panel(slide, x, y, w, h, fill=None, accent=None):
    sh = slide.shapes.add_shape(1, x, y, w, h)
    sh.fill.solid(); sh.fill.fore_color.rgb = fill or C["panel"]
    sh.line.color.rgb = C["rule"]; sh.line.width = Pt(0.75)
    sh.shadow.inherit = False
    if accent:
        bar = slide.shapes.add_shape(1, x, y, w, Inches(0.045))
        bar.fill.solid(); bar.fill.fore_color.rgb = accent
        bar.line.fill.background(); bar.shadow.inherit = False
    return sh


def table(slide, x, y, w, rows, col_w, header=None, row_h=Inches(0.42)):
    """Minimal table: header row + body, hairline separators, no grid."""
    yy = y
    if header:
        cx = x
        for text, cw in zip(header, col_w):
            tf = tb(slide, cx, yy, cw, Inches(0.26))
            run(tf.paragraphs[0], text.upper(), 9.5, C["faint"], bold=True)
            cx += cw
        yy += Inches(0.30)
        rule(slide, yy); yy += Inches(0.12)
    for r in rows:
        cx = x
        for cell, cw in zip(r, col_w):
            text, *style = cell if isinstance(cell, tuple) else (cell,)
            color = style[0] if style else C["ink"]
            bold = style[1] if len(style) > 1 else False
            size = style[2] if len(style) > 2 else 13
            tf = tb(slide, cx, yy, cw, row_h, anchor=MSO_ANCHOR.TOP)
            run(tf.paragraphs[0], text, size, color, bold=bold)
            cx += cw
        yy += row_h
        rule(slide, yy - Inches(0.08))
    return yy


# ───────────────────────── 1 · FERMAT ─────────────────────────
s = prs.slides.add_slide(BLANK)
eyebrow(s, "The dataset", C["accent"])
title(s, "FERMAT: handwritten math,\nerrors planted on purpose")

s.shapes.add_picture(f"{CASES}/07_qwen7b_confidently_wrong_grading/image.jpg",
                     M, Inches(2.25), height=Inches(3.9))

x2 = Inches(6.6)
tf = tb(s, x2, Inches(2.25), Inches(6.1), Inches(1.6))
for i, (a, b) in enumerate([("~2,200", " photographed solutions"),
                            ("85%", " contain a deliberate error"),
                            ("Question + answer", " both in the image")]):
    p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
    p.space_after = Pt(9)
    run(p, a, 16, C["ink"], bold=True); run(p, b, 16, C["soft"])

tf = tb(s, x2, Inches(3.95), Inches(6.1), Inches(0.3))
run(tf.paragraphs[0], "ONLY FERMAT HAS CORRECT ANSWERS TOO", 10.5, C["faint"], bold=True)
table(s, x2, Inches(4.35), Inches(6.1),
      rows=[["FERMAT", ("15% clean", C["good"], True)],
            ["ErrorRadar", ("none", C["bad"], True)],
            ["ScratchMath", ("none", C["bad"], True)]],
      col_w=[Inches(3.2), Inches(2.9)])

tf = tb(s, x2, Inches(5.95), Inches(6.1), Inches(0.6))
run(tf.paragraphs[0], "No clean items → no balanced set → the finding on slide 4 "
    "cannot be measured at all.", 12.5, C["soft"], italic=True)
slide_num(s, 1)

# ───────────────────── 2 · perception works ─────────────────────
s = prs.slides.add_slide(BLANK)
eyebrow(s, "Result — confirmed", C["good"])
title(s, "Disagreement predicts error")

tf = tb(s, M, Inches(1.75), Inches(6), Inches(0.4))
run(tf.paragraphs[0], "Read the same page 5 times. Measure the disagreement.",
    15, C["soft"])

table(s, M, Inches(2.55), Inches(6.6),
      header=["", "AUROC"],
      rows=[[("Self-disagreement", C["ink"], True), ("0.835", C["good"], True, 15)],
            ["…after every artifact cut", ("0.796", C["good"], True, 15)],
            ["Model's own confidence", ("0.537", C["bad"], True, 15)]],
      col_w=[Inches(4.3), Inches(2.3)], row_h=Inches(0.5))

panel(s, Inches(7.5), Inches(2.4), Inches(5.2), Inches(2.4), accent=C["good"])
tf = tb(s, Inches(7.85), Inches(2.75), Inches(4.5), Inches(1.9))
p = tf.paragraphs[0]
run(p, "All 5 readings disagree?", 19, C["ink"], font=SERIF, bold=True)
p = tf.add_paragraph(); p.space_before = Pt(8)
run(p, "Wrong 57 times out of 61.", 19, C["good"], font=SERIF, bold=True)
p = tf.add_paragraph(); p.space_before = Pt(12)
run(p, "A usable abstention rule. No training, no labels.", 12.5, C["soft"])
slide_num(s, 2)

# ─────────────── 3 · the four quadrants ───────────────
s = prs.slides.add_slide(BLANK)
eyebrow(s, "Evidence — four real pages", C["accent"])
title(s, "Where it works, and the blind spot", size=30)

quad = [
    ("01_qwen3b_perception_low_entropy_correct", "Agrees  ·  H = 0.00", "Correct", C["good"]),
    ("02_qwen3b_perception_high_entropy_wrong",  "Disagrees  ·  H = 1.61", "Caught", C["good"]),
    ("03_qwen3b_perception_high_entropy_correct","Disagrees  ·  H = 1.61", "False alarm", C["warn"]),
    ("04_qwen3b_perception_low_entropy_wrong",   "Agrees  ·  H = 0.00", "Blind spot", C["bad"]),
]
cw, ch = Inches(3.02), Inches(3.05)
# Fit each page inside a fixed box rather than setting width: these four
# images run from 1600x878 landscape to 1298x1600 portrait, and sizing by
# width alone makes the tall ones ~3.4in high, overflowing the panel and
# colliding with the caption. Scale to whichever dimension binds, then
# centre horizontally so the row stays visually even.
BOX_W, BOX_H = cw - Inches(0.3), Inches(2.05)
for i, (case, lab, verdict, col) in enumerate(quad):
    cx = M + (cw + Inches(0.14)) * i
    panel(s, cx, Inches(1.95), cw, ch, fill=C["white"], accent=col)

    with Image.open(f"{CASES}/{case}/image.jpg") as im:
        iw, ih = im.size
    scale = min(BOX_W / iw, BOX_H / ih)
    pw, ph = int(iw * scale), int(ih * scale)
    s.shapes.add_picture(f"{CASES}/{case}/image.jpg",
                         cx + (cw - pw) // 2, Inches(2.12) + (BOX_H - ph) // 2,
                         width=pw, height=ph)

    tf = tb(s, cx + Inches(0.15), Inches(4.32), cw - Inches(0.3), Inches(0.6))
    run(tf.paragraphs[0], lab, 10.5, C["soft"])
    p = tf.add_paragraph(); p.space_before = Pt(2)
    run(p, verdict, 13, col, bold=True)

panel(s, M, Inches(5.32), W - 2 * M, Inches(1.35), accent=C["bad"])
tf = tb(s, M + Inches(0.35), Inches(5.62), Inches(11.8), Inches(0.9))
run(tf.paragraphs[0], "Page 1 and page 4 look identical to the method — both entropy zero.",
    19, C["ink"], font=SERIF, bold=True)
p = tf.add_paragraph(); p.space_before = Pt(7)
run(p, "One is right, one is wrong. Confident errors are invisible to it.",
    13.5, C["soft"])
slide_num(s, 3)

# ─────────────── 4 · pooled vs stratified ───────────────
s = prs.slides.add_slide(BLANK)
eyebrow(s, "Method warning — the key finding", C["warn"])
title(s, "The average hid the finding")

tf = tb(s, M, Inches(1.78), Inches(11.5), Inches(0.4))
run(tf.paragraphs[0], "Grading task. Same 300 items, split by whether an error was really there.",
    15, C["soft"])

table(s, M, Inches(2.6), Inches(12),
      header=["Scored on", "AUROC", ""],
      rows=[[("Everything pooled", C["ink"], True), ("0.520", C["bad"], True, 16),
             ("looks like chance", C["bad"], False, 13)],
            [("Items with an error", C["ink"], True), ("0.801", C["good"], True, 16),
             ("works well", C["good"], False, 13)],
            [("Items that are correct", C["ink"], True), ("0.280", C["good"], True, 16),
             ("works backwards", C["good"], False, 13)]],
      col_w=[Inches(4.6), Inches(2.2), Inches(5.2)], row_h=Inches(0.62))

panel(s, M, Inches(5.05), W - 2 * M, Inches(1.5), accent=C["bad"])
tf = tb(s, M + Inches(0.35), Inches(5.35), Inches(11.8), Inches(1.0))
run(tf.paragraphs[0], "0.80 and 0.28 average out to 0.52.", 23, C["ink"],
    font=SERIF, bold=True)
p = tf.add_paragraph(); p.space_before = Pt(8)
run(p, "Both halves are real. Pooling reports neither. "
       "Only visible because FERMAT has correct items.", 13.5, C["soft"])
slide_num(s, 4)

# ─────────────── 5 · replication ───────────────
s = prs.slides.add_slide(BLANK)
eyebrow(s, "Replication", C["good"])
title(s, "Holds across three model families")

table(s, M, Inches(2.2), Inches(6.4),
      header=["Model", "AUROC"],
      rows=[["Qwen2.5-VL-3B", ("0.854", C["good"], True, 15)],
            ["Qwen2.5-VL-7B", ("0.801", C["good"], True, 15)],
            ["LLaVA-NeXT-7B", ("0.775", C["good"], True, 15)]],
      col_w=[Inches(4.0), Inches(2.4)], row_h=Inches(0.52))

tf = tb(s, M, Inches(4.15), Inches(6.4), Inches(0.5))
run(tf.paragraphs[0], "Confidence intervals overlap. Not a Qwen quirk.", 13.5, C["soft"])

panel(s, Inches(7.5), Inches(2.1), Inches(5.2), Inches(3.0), accent=C["bad"])
tf = tb(s, Inches(7.85), Inches(2.45), Inches(4.5), Inches(2.5))
run(tf.paragraphs[0], "A 4th model looked best", 12, C["faint"], bold=True)
p = tf.add_paragraph(); p.space_before = Pt(10)
run(p, "0.915", 30, C["bad"], font=SERIF, bold=True)
run(p, "   →   ", 20, C["faint"])
run(p, "0.556", 30, C["ink"], font=SERIF, bold=True)
p = tf.add_paragraph(); p.space_before = Pt(12)
run(p, "InternVL3. Under the same robustness check, it collapsed to chance — "
       "it was producing incoherent output on 2/3 of items.", 13, C["soft"])
slide_num(s, 5)

# ─────────────── 6 · open decision ───────────────
s = prs.slides.add_slide(BLANK)
eyebrow(s, "Open decision", C["accent"])
title(s, "One question")

panel(s, M, Inches(2.0), Inches(5.95), Inches(2.75), accent=C["good"])
tf = tb(s, M + Inches(0.32), Inches(2.3), Inches(5.3), Inches(2.3))
run(tf.paragraphs[0], "SOLID", 11, C["good"], bold=True)
for t in ["Perception signal 0.835, robust",
          "Abstention rule 57/61",
          "Confirmed on 3 model families",
          "Pooled-vs-split: a method contribution"]:
    p = tf.add_paragraph(); p.space_before = Pt(9)
    run(p, "— " + t, 13.5, C["ink"])

panel(s, Inches(6.95), Inches(2.0), Inches(5.75), Inches(2.75), accent=C["warn"])
tf = tb(s, Inches(7.27), Inches(2.3), Inches(5.1), Inches(2.3))
run(tf.paragraphs[0], "LIMITED", 11, C["warn"], bold=True)
for t in ["Everything sits on one dataset",
          "Both alternatives are 100% error items",
          "…so they cannot test the split at all",
          "On one, the model said “error” while\n    saying it could not read the image"]:
    p = tf.add_paragraph(); p.space_before = Pt(9)
    run(p, "— " + t, 13.5, C["ink"])

panel(s, M, Inches(5.05), W - 2 * M, Inches(1.5), accent=C["accent"])
tf = tb(s, M + Inches(0.35), Inches(5.35), Inches(11.8), Inches(1.0))
run(tf.paragraphs[0], "Accept one dataset as a limitation — or label correct items ourselves?",
    21, C["ink"], font=SERIF, bold=True)
p = tf.add_paragraph(); p.space_before = Pt(8)
run(p, "~3 weeks to submission. All experiments done; remaining work is the write-up.",
    13.5, C["soft"])
slide_num(s, 6)

out = "/private/tmp/claude-501/-Users-sepehrmaleki-Documents-spring-2026-uncertainty-math-vlm/d55af82e-e184-4387-a524-130d2a77780c/scratchpad/deck/fermat_findings.pptx"
prs.save(out)
import os
print(f"saved {out}  ({os.path.getsize(out)/1024/1024:.2f} MB, {len(prs.slides.__iter__.__self__._sldIdLst)} slides)")
