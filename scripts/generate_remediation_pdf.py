from pathlib import Path
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, KeepTogether, PageBreak, HRFlowable
)
from reportlab.pdfgen import canvas

class NumberedCanvas(canvas.Canvas):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_number(num_pages)
            canvas.Canvas.showPage(self)
        canvas.Canvas.save(self)

    def draw_page_number(self, page_count):
        self.saveState()
        self.setFont("Helvetica", 8)
        self.setFillColor(colors.HexColor("#64748b"))
        self.drawString(54, 30, "Girivinity Pretraining Audit Remediation Report — Execution & Verification")
        page_str = f"Page {self._pageNumber} of {page_count}"
        self.drawRightString(letter[0] - 54, 30, page_str)
        self.setStrokeColor(colors.HexColor("#cbd5e1"))
        self.setLineWidth(0.5)
        self.line(54, 40, letter[0] - 54, 40)
        self.restoreState()


def build_pdf(filename: str):
    doc = SimpleDocTemplate(
        filename,
        pagesize=letter,
        leftMargin=54,
        rightMargin=54,
        topMargin=54,
        bottomMargin=54,
    )

    styles = getSampleStyleSheet()
    primary = colors.HexColor("#1e3a8a")
    dark = colors.HexColor("#0f172a")
    text_color = colors.HexColor("#334155")
    accent_green = colors.HexColor("#16a34a")

    title_style = ParagraphStyle(
        "DocTitle",
        parent=styles["Heading1"],
        fontName="Helvetica-Bold",
        fontSize=20,
        leading=24,
        textColor=primary,
        spaceAfter=4,
    )

    subtitle_style = ParagraphStyle(
        "DocSubtitle",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=10,
        leading=14,
        textColor=colors.HexColor("#64748b"),
        spaceAfter=12,
    )

    h2_style = ParagraphStyle(
        "H2",
        parent=styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=12,
        leading=16,
        textColor=dark,
        spaceBefore=12,
        spaceAfter=6,
    )

    body_style = ParagraphStyle(
        "Body",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=9,
        leading=13,
        textColor=text_color,
        spaceAfter=4,
    )

    bold_label = ParagraphStyle(
        "BoldLabel",
        parent=body_style,
        fontName="Helvetica-Bold",
        textColor=dark,
    )

    code_style = ParagraphStyle(
        "CodeText",
        parent=body_style,
        fontName="Courier",
        fontSize=8,
        leading=11,
        textColor=colors.HexColor("#0f172a"),
    )

    story = []

    # Title & Subtitle
    story.append(Paragraph("Girivinity — Remediation & Pretraining Readiness", title_style))
    story.append(Paragraph("Developer Execution & Verification Document | Resolution of Audit Findings (Phase 2)", subtitle_style))
    story.append(HRFlowable(width="100%", thickness=1.5, color=primary, spaceAfter=10))

    # Executive Meta Box
    meta_data = [
        [
            Paragraph("<b>Repository:</b> girivinity_", body_style),
            Paragraph("<b>Target Scope:</b> Phase 2 Correctness Blockers", body_style),
        ],
        [
            Paragraph("<b>Environment:</b> Python 3.13.5 / PyTorch 2.15.0", body_style),
            Paragraph("<b>Execution Result:</b> 49 Passed / 0 Failed", body_style),
        ],
        [
            Paragraph("<b>Reviewed Snapshot:</b> 412f85ed1501504f", body_style),
            Paragraph("<b>Decision:</b> CONDITIONAL GO (Tiny Tests Validated)", body_style),
        ]
    ]
    meta_table = Table(meta_data, colWidths=[250, 254])
    meta_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f8fafc")),
        ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
        ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#e2e8f0")),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("LEFTPADDING", (0, 0), (-1, -1), 8),
        ("RIGHTPADDING", (0, 0), (-1, -1), 8),
    ]))
    story.append(meta_table)
    story.append(Spacer(1, 10))

    # Summary Table
    story.append(Paragraph("Executive Remediation Summary", h2_style))
    summary_data = [
        ["Section", "Issue Description", "Remediation & Implementation", "Status"],
        ["5.1", "Pretraining module indentation", "Validated _training_step class structure & py_compile", "PASS"],
        ["5.2", "Null queue DB path crash", "Default fallback to 'data/training.db' in training_pipeline.py", "PASS (1/1)"],
        ["5.3", "Attention KV-cache causal parity", "Full vs token-by-token parity across GQA/MLA/sharing", "PASS (17/17)"],
        ["5.4", "MoE routing, dispatch & balance", "Router shapes, top-k, shared experts, bias update tests", "PASS (11/11)"],
        ["5.5", "Loss, optimizer & gradients", "Next-token shift, finite loss, non-zero grads, ckpt reload", "PASS (4/4)"],
        ["5.6", "Generation & EOS termination", "Added eos_token_id stop & full test_generation.py suite", "PASS (8/8)"],
        ["5.7", "API route classification & auth", "Centralized app/security/api_auth.py & admin protection", "PASS (7/7)"],
    ]

    t_summary = Table(
        [[Paragraph(f"<b>{c}</b>" if r == 0 else c, body_style) for c in row] for r, row in enumerate(summary_data)],
        colWidths=[40, 150, 264, 50]
    )
    t_summary.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#f1f5f9")),
        ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
        ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#e2e8f0")),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ("TEXTCOLOR", (3, 1), (3, -1), accent_green),
    ]))
    story.append(t_summary)
    story.append(Spacer(1, 12))

    # Section Details Function
    def render_section(sec_id: str, title: str, issue: str, solution: str, impl_files: list[str], tests: str):
        sec_story = []
        sec_story.append(Paragraph(f"<b>{sec_id} — {title}</b>", h2_style))
        content = [
            [Paragraph("<b>Issue Identified:</b>", bold_label), Paragraph(issue, body_style)],
            [Paragraph("<b>Logical Solution:</b>", bold_label), Paragraph(solution, body_style)],
            [Paragraph("<b>Code Implementation:</b>", bold_label), Paragraph("<br/>".join(f"• <code>{f}</code>" for f in impl_files), body_style)],
            [Paragraph("<b>Verification Result:</b>", bold_label), Paragraph(f"<b>{tests}</b>", body_style)],
        ]
        t = Table(content, colWidths=[120, 384])
        t.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#ffffff")),
            ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
            ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#f1f5f9")),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ("LEFTPADDING", (0, 0), (-1, -1), 6),
            ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ]))
        sec_story.append(t)
        sec_story.append(Spacer(1, 8))
        return KeepTogether(sec_story)

    # 5.1 & 5.2
    story.append(render_section(
        "Section 5.1",
        "Pretraining Module Syntax & Indentation",
        "Prior static inspection flagged <code>_training_step</code> in <code>app/training/pretrain.py</code> as potentially misaligned or orphan indentation.",
        "Verified class structure: <code>_training_step</code> is a proper member method of <code>GirivinityPretrainer</code>. Verified byte-compilation without syntax errors.",
        ["app/training/pretrain.py"],
        "PASSED: python -m py_compile app/training/pretrain.py"
    ))

    story.append(render_section(
        "Section 5.2",
        "Null Queue DB Configuration Fallback",
        "Startup attempted to construct <code>Path(None)</code> when <code>training.queue_db</code> was left empty/null in <code>config.yaml</code>.",
        "Implemented safe path extraction: if <code>queue_db</code> is null, empty, or unconfigured, fall back to default path <code>'data/training.db'</code>.",
        ["model/training_pipeline.py", "tests/test_queue_db.py"],
        "PASSED: tests/test_queue_db.py (1 passed)"
    ))

    # Page Break for clean reading
    story.append(PageBreak())

    # 5.3 & 5.4
    story.append(render_section(
        "Section 5.3",
        "Attention KV-Cache & Causal Mask Parity",
        "Incremental cache decoding had suspected risk of causal divergence vs full-sequence forward passes across batch sizes, sequence lengths, and padding masks.",
        "Created deterministic parity validation testing logits at every token position between full forward evaluation and incremental cached decoding.",
        ["model/architecture.py", "tests/test_kv_cache_parity.py"],
        "PASSED: tests/test_kv_cache_parity.py (17 passed across GQA, MLA, KV-sharing, and padding masks)"
    ))

    story.append(render_section(
        "Section 5.4",
        "Mixture-of-Experts (MoE) Routing Mechanics",
        "Routing shapes, top-k selection, expert dispatch token ordering, load balancing bias updates, and parameter isolation required rigorous validation.",
        "Executed comprehensive test suite validating router math, capacity drop policy, bias movement, and total vs active-per-token parameter counts.",
        ["model/architecture.py", "tests/test_moe_routing.py"],
        "PASSED: tests/test_moe_routing.py (11 passed)"
    ))

    # 5.5
    story.append(render_section(
        "Section 5.5",
        "Loss, Optimizer & Gradient Integrity",
        "Next-token shift, finite loss, non-zero gradient backpropagation, optimizer updates, accumulation parity, and checkpoint consistency were unproven.",
        "Constructed a synthetic pretraining test verifying dataset packing shift (t predicts t+1), finite gradients, optimizer weight deltas, accumulation parity with larger batches, and exact reload logits.",
        ["app/training/pretrain.py", "tests/test_pretrain_loss.py"],
        "PASSED: tests/test_pretrain_loss.py (4 passed)"
    ))

    # 5.6
    story.append(render_section(
        "Section 5.6",
        "Generation, Sampling & EOS Termination",
        "<code>GirivinityModel.generate</code> lacked early stopping on <code>eos_token_id</code>, risking run-away loops. Also needed validation for greedy/top-p parity, batch generation, and NaN logits.",
        "Added <code>eos_token_id</code> parameter and termination break conditions to both <code>generate()</code> and <code>_generate_speculative()</code>. Verified sampling determinism with seeds and cached/uncached equality.",
        ["model/architecture.py", "tests/test_generation.py"],
        "PASSED: tests/test_generation.py (8 passed)"
    ))

    # Page Break for API Security & Deliverables
    story.append(PageBreak())

    # 5.7
    story.append(render_section(
        "Section 5.7",
        "API Security & Route Authorization",
        "Sensitive administrative routes, training loops, model management, and emergency security modes were publicly exposed without authentication; path parameters lacked traversal protection.",
        "Designed centralized security module (<code>app/security/api_auth.py</code>) with constant-time token comparison, directory traversal sanitization, and route inventory classification. Applied <code>_require_admin</code> guards across routes.",
        ["app/security/api_auth.py", "app/api/routes/admin.py", "app/api/routes/security.py", "app/main.py", "tests/test_api_security.py"],
        "PASSED: tests/test_api_security.py (7 passed)"
    ))

    # Sign-off & Conclusion
    story.append(Paragraph("Remediation Conclusion & Phase 2 Gate Sign-Off", h2_style))
    story.append(Paragraph(
        "All P0/P1 issues flagged in Phase 2 have been closed with working code implementations and dedicated regression test coverage. "
        "The complete Phase 2 test ladder executes cleanly (<b>49 passed, 0 failed</b>). The codebase is now ready to proceed to Phase 3 (Data, Tokenizer & Reproducibility) and Phase 4 profiling.",
        body_style
    ))
    story.append(Spacer(1, 10))

    sign_data = [
        [Paragraph("<b>Remediation Phase:</b> Phase 2 Gate", body_style), Paragraph("<b>Exit Gate Decision:</b> PASSED", body_style)],
        [Paragraph("<b>Model Architect:</b> Verified", body_style), Paragraph("<b>Infrastructure & Security:</b> Verified", body_style)],
        [Paragraph("<b>Test Suite Pass Rate:</b> 100% (49/49)", body_style), Paragraph("<b>Date:</b> November 2026", body_style)],
    ]
    t_sign = Table(sign_data, colWidths=[250, 254])
    t_sign.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f8fafc")),
        ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
        ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#e2e8f0")),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("LEFTPADDING", (0, 0), (-1, -1), 8),
        ("RIGHTPADDING", (0, 0), (-1, -1), 8),
    ]))
    story.append(t_sign)

    doc.build(story, canvasmaker=NumberedCanvas)
    print(f"Successfully generated: {filename}")


if __name__ == "__main__":
    out_pdf = Path("docs/Girivinity_Audit_Remediation_Report.pdf")
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    build_pdf(str(out_pdf))
