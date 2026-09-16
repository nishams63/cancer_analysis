#!/usr/bin/env python3
"""
Generates a comprehensive, publication-grade executive summary PDF for the
Personalized Precision Medicine for Oncology Treatment Optimization project
(https://github.com/nishams63/cancer_analysis).
"""

import os
from pathlib import Path
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT, TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    PageBreak, KeepTogether
)
from reportlab.pdfgen import canvas

# Color Palette
navy = colors.HexColor('#0F172A')       # Dark Navy
blue = colors.HexColor('#1D4ED8')       # Accent Blue
sky = colors.HexColor('#0284C7')        # Sky Blue
teal = colors.HexColor('#0D9488')       # Teal
slate = colors.HexColor('#334155')      # Body slate
light_slate = colors.HexColor('#64748B')# Muted slate
bg_card = colors.HexColor('#F8FAFC')    # Card background
border_color = colors.HexColor('#E2E8F0')# Border color
emerald = colors.HexColor('#059669')    # Success green

class NumberedCanvas(canvas.Canvas):
    """Two-pass canvas to dynamically compute and print total page numbers."""
    def __init__(self, *args, **kwargs):
        super(NumberedCanvas, self).__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_decorations(num_pages)
            super(NumberedCanvas, self).showPage()
        super(NumberedCanvas, self).save()

    def draw_page_decorations(self, page_count):
        self.saveState()
        self.setFont("Helvetica", 8)
        self.setFillColor(light_slate)
        
        # Header (pages after cover)
        if self._pageNumber > 1:
            self.drawString(40, A4[1] - 28, "Oncology Treatment Optimization — Project Summary & Architecture Analysis")
            self.drawRightString(A4[0] - 40, A4[1] - 28, "https://github.com/nishams63/cancer_analysis")
            self.setStrokeColor(border_color)
            self.setLineWidth(0.6)
            self.line(40, A4[1] - 33, A4[0] - 40, A4[1] - 33)

        # Footer (all pages)
        self.setStrokeColor(border_color)
        self.setLineWidth(0.6)
        self.line(40, 36, A4[0] - 40, 36)
        self.drawString(40, 24, "Confidential — Personalized Precision Oncology AI Platform")
        page_str = f"Page {self._pageNumber} of {page_count}"
        self.drawRightString(A4[0] - 40, 24, page_str)
        self.restoreState()


def build_pdf(out_path: str):
    doc = SimpleDocTemplate(
        out_path,
        pagesize=A4,
        leftMargin=38,
        rightMargin=38,
        topMargin=42,
        bottomMargin=42
    )

    styles = getSampleStyleSheet()

    # Typography Styles
    title_style = ParagraphStyle(
        name='DocTitle',
        fontName='Helvetica-Bold',
        fontSize=20,
        leading=25,
        textColor=navy,
        spaceAfter=4
    )
    subtitle_style = ParagraphStyle(
        name='DocSubTitle',
        fontName='Helvetica',
        fontSize=10,
        leading=14,
        textColor=sky,
        spaceAfter=12
    )
    h1_style = ParagraphStyle(
        name='SectionH1',
        fontName='Helvetica-Bold',
        fontSize=12.5,
        leading=16,
        textColor=navy,
        spaceBefore=10,
        spaceAfter=5,
        keepWithNext=True
    )
    h2_style = ParagraphStyle(
        name='SectionH2',
        fontName='Helvetica-Bold',
        fontSize=10,
        leading=13.5,
        textColor=blue,
        spaceBefore=7,
        spaceAfter=3,
        keepWithNext=True
    )
    body_style = ParagraphStyle(
        name='MainBody',
        fontName='Helvetica',
        fontSize=8.3,
        leading=11.6,
        textColor=slate,
        spaceAfter=3.5,
        alignment=TA_JUSTIFY
    )
    bullet_style = ParagraphStyle(
        name='BulletItem',
        fontName='Helvetica',
        fontSize=8.1,
        leading=11.3,
        textColor=slate,
        leftIndent=10,
        spaceAfter=2
    )
    table_hdr = ParagraphStyle(
        name='TableHdr',
        fontName='Helvetica-Bold',
        fontSize=8,
        leading=10,
        textColor=navy,
        alignment=TA_LEFT
    )
    table_cell = ParagraphStyle(
        name='TableCell',
        fontName='Helvetica',
        fontSize=7.6,
        leading=9.8,
        textColor=slate,
        alignment=TA_LEFT
    )
    table_cell_bold = ParagraphStyle(
        name='TableCellBold',
        fontName='Helvetica-Bold',
        fontSize=7.6,
        leading=9.8,
        textColor=navy,
        alignment=TA_LEFT
    )
    code_style = ParagraphStyle(
        name='CodeStyle',
        fontName='Courier',
        fontSize=7.5,
        leading=10,
        textColor=colors.HexColor('#0F766E')
    )

    story = []

    # ==================== PAGE 1 ====================
    story.append(Paragraph("Personalized Precision Medicine for Oncology Treatment Optimization", title_style))
    story.append(Paragraph("Comprehensive Technical Architecture & Engineering Deliverables Summary | Git Repository Analysis", subtitle_style))
    
    meta_table_data = [
        [
            Paragraph("<b>Repository:</b> <font color='#1d4ed8'>https://github.com/nishams63/cancer_analysis</font>", table_cell),
            Paragraph("<b>Status:</b> <font color='#059669'><b>5 Stages Deployed & Verified</b></font>", table_cell),
            Paragraph("<b>Date:</b> September 2026", table_cell)
        ],
        [
            Paragraph("<b>Core Frameworks:</b> PyTorch, XGBoost, Transformers, QLoRA, FastAPI, ReportLab, Three.js", table_cell),
            Paragraph("<b>GenAI Models:</b> LLaMA-3.2-11B, LLaMA-3.1-70B, BioMistral-7B, Phi-3, BioLinkBERT", table_cell),
            Paragraph("<b>Test Suite:</b> 116+ Unit Tests (100% Pass Rate)", table_cell)
        ]
    ]
    t_meta = Table(meta_table_data, colWidths=[205, 185, 129])
    t_meta.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), bg_card),
        ('BOX', (0, 0), (-1, -1), 0.7, border_color),
        ('INNERGRID', (0, 0), (-1, -1), 0.4, border_color),
        ('TOPPADDING', (0, 0), (-1, -1), 3.5),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 3.5),
        ('LEFTPADDING', (0, 0), (-1, -1), 5),
        ('RIGHTPADDING', (0, 0), (-1, -1), 5),
    ]))
    story.append(t_meta)
    story.append(Spacer(1, 6))

    # --- Section 1: Executive Summary ---
    story.append(Paragraph("1. Executive Summary & Vision", h1_style))
    story.append(Paragraph(
        "The <b>Personalized Precision Medicine for Oncology Treatment Optimization</b> platform is an autonomous, "
        "multi-stage clinical AI system designed to improve patient survival, mitigate severe antineoplastic toxicities, "
        "predict cancer recurrence, and stress-test clinical AI safety using cutting-edge Generative AI. "
        "The repository embodies an enterprise-grade machine learning and deep learning architecture that spans "
        "classical tabular predictive modeling (Stage 1), multimodal imaging and longitudinal ctDNA sequencing (Stage 2), "
        "negation-aware clinical NLP (Stage 3), fine-tuned Small Language Models (Stage 4), and an advanced GenAI Synthetic "
        "Stress-Test Engine with RAG guidance and interactive dashboard hosting (Stage 5).",
        body_style
    ))
    story.append(Paragraph(
        "Each stage was systematically constructed following strict production MLOps standards: zero-leakage data engineering, "
        "mathematically rigorous exploratory data analysis (EDA), multi-model benchmarking, independent locked-test evaluation with "
        "clinical subgroup audits, containerized FastAPI services, and programmatic Definition-of-Done (DoD) verifications.",
        body_style
    ))

    # Master Stages Summary Table
    stages_data = [
        [Paragraph("Stage", table_hdr), Paragraph("Focus & Discipline", table_hdr), Paragraph("Key Technologies & Models", table_hdr), Paragraph("Primary Deliverable & Impact", table_hdr)],
        [
            Paragraph("<b>Stage 1</b>", table_cell_bold),
            Paragraph("Classical Tabular ML Toxicity Prediction", table_cell),
            Paragraph("RandomForest, XGBoost, CatBoost, Cox-PH, Optuna, Joblib", table_cell),
            Paragraph("Candidate V4 model; predicts severe adverse drug toxicity across 26 cohorts; FastAPI REST service.", table_cell)
        ],
        [
            Paragraph("<b>Stage 2</b>", table_cell_bold),
            Paragraph("Multimodal Deep Learning Progression", table_cell),
            Paragraph("Vision CNNs (DenseNet), LSTMs, Transformers, MIL Attention", table_cell),
            Paragraph("Fuses histopathology tiles with longitudinal ctDNA time-series for recurrence risk.", table_cell)
        ],
        [
            Paragraph("<b>Stage 3</b>", table_cell_bold),
            Paragraph("Clinical NLP & Concept Extraction", table_cell),
            Paragraph("BioLinkBERT, DeBERTa, NegEx Scoping, Rule-based Heuristics", table_cell),
            Paragraph("Extracts clinical entities, resolves negation leakage, triages urgent toxicity (Macro F1 0.7557).", table_cell)
        ],
        [
            Paragraph("<b>Stage 4</b>", table_cell_bold),
            Paragraph("SLM Fine-Tuning & Clinical Guidance", table_cell),
            Paragraph("BioMistral-7B, Clinical-LLaMA-3, QLoRA, GGUF, llama.cpp", table_cell),
            Paragraph("Parameter-efficient fine-tuning; 6-gate clinical safety firewall; 100% offline local CPU inference.", table_cell)
        ],
        [
            Paragraph("<b>Stage 5</b>", table_cell_bold),
            Paragraph("GenAI Synthetic Stress-Test Engine", table_cell),
            Paragraph("Monte Carlo Sampler, NVIDIA LLaMA-3.2-11B/3.1-70B, RAG Vector Store", table_cell),
            Paragraph("Hybrid statistical + LLM generation; 15 blind spots (BS01–BS15); interactive web dashboard.", table_cell)
        ]
    ]
    t_stages = Table(stages_data, colWidths=[45, 138, 160, 176])
    t_stages.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), bg_card),
        ('BOX', (0, 0), (-1, -1), 0.7, border_color),
        ('INNERGRID', (0, 0), (-1, -1), 0.4, border_color),
        ('TOPPADDING', (0, 0), (-1, -1), 3),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
        ('LEFTPADDING', (0, 0), (-1, -1), 4.5),
        ('RIGHTPADDING', (0, 0), (-1, -1), 4.5),
    ]))
    story.append(t_stages)
    story.append(Spacer(1, 6))

    # --- Section 2: Detailed Engineering Breakdown ---
    story.append(Paragraph("2. Detailed Engineering Breakdown by Stage", h1_style))

    # Stage 1
    story.append(Paragraph("Stage 1: Classical Machine Learning for Adverse Drug Toxicity Prediction", h2_style))
    story.append(Paragraph("• <b>Data Engineering:</b> Ingested, cleaned, and standardized multi-center oncology patient records into <code>master_patient_dataset.csv</code>. Handled missingness through domain-guided imputation and established schema validations.", bullet_style))
    story.append(Paragraph("• <b>Exploratory Data Analysis (EDA):</b> Mapped high-risk biomarker interactions (e.g. baseline creatinine vs platinum clearance), class imbalance across toxicity grades, and generated correlation heatmaps.", bullet_style))
    story.append(Paragraph("• <b>Model Architecture & Candidate V4:</b> Benchmarked Random Forest, CatBoost, and XGBoost against baseline heuristics. Regularized the champion <b>Candidate V4</b> model to prevent overfitting and serialized via <code>model.joblib</code>.", bullet_style))
    story.append(Paragraph("• <b>Evaluation & Subgroups:</b> Evaluated on a strictly locked test set using 95% bootstrap confidence intervals across 26 distinct demographic and clinical cohorts, proving consistent recall across age and gender.", bullet_style))
    story.append(Paragraph("• <b>Production Integration:</b> Packaged as a high-throughput FastAPI microservice exposing <code>/health</code>, <code>/predict</code>, and <code>/predict/batch</code> with full Pydantic request validation.", bullet_style))

    # ==================== PAGE 2 ====================
    story.append(PageBreak())

    # Stage 2
    story.append(Paragraph("Stage 2: Deep Learning Multimodal Progression & Recurrence Prediction", h2_style))
    story.append(Paragraph("• <b>Multimodal Data Pipeline:</b> Preprocessed gigapixel synthetic pathology tiles and paired them with longitudinal serial ctDNA VAF and tumor marker (CEA, CA-125) time-series sequences.", bullet_style))
    story.append(Paragraph("• <b>Spatial Deep Learning:</b> Deployed Convolutional Neural Networks (DenseNet/ResNet) and Multiple Instance Learning (MIL) attention architectures to extract spatial tumor microenvironment features.", bullet_style))
    story.append(Paragraph("• <b>Temporal Sequence Models:</b> Built Bidirectional LSTMs and Temporal Transformers capturing velocity and acceleration spikes in circulating tumor DNA before radiological visibility.", bullet_style))
    story.append(Paragraph("• <b>Multimodal Late Fusion API:</b> Implemented a late-fusion gating layer combining spatial and temporal embeddings into a single calibrated risk index, wrapped in a dedicated inference API and clinical alert engine.", bullet_style))

    # Stage 3
    story.append(Paragraph("Stage 3: Clinical NLP & Decision Support Engineering", h2_style))
    story.append(Paragraph("• <b>Research Dataset Pipeline:</b> Generated <code>clinical_nlp_dataset_v1.parquet</code> with automated de-identification (PII scrub), semantic deduplication, and zero-leakage cross-validation splits.", bullet_style))
    story.append(Paragraph("• <b>20-Section EDA:</b> Profiled clinical vocabulary, entity distribution, sentence syntax, and created 16 analytical figures in <code>eda_summary.json</code> and exploratory notebooks.", bullet_style))
    story.append(Paragraph("• <b>Clinical NLP Modeling:</b> Built negation-aware extractors using BioLinkBERT, DeBERTa, and hybrid clinical heuristics, achieving a Macro F1 score of <b>0.7557</b> in acute clinical concept triage.", bullet_style))
    story.append(Paragraph("• <b>Diagnostic Hardening:</b> Solved negation scope leakage across contrastive conjunctions ('Denies chest pain, however rapid pulse observed') to ensure accurate symptom triage.", bullet_style))

    # Stage 4
    story.append(Paragraph("Stage 4: Small Language Model (SLM) Fine-Tuning & Multi-Agent Decision Support", h2_style))
    story.append(Paragraph("• <b>4 Engineering Submodules:</b> Structured across Data Engineer, EDA Engineer, SLM Engineer, and Evaluation Engineer.", bullet_style))
    story.append(Paragraph("• <b>QLoRA Fine-Tuning:</b> Fine-tuned open-weights models (BioMistral-7B, Clinical-LLaMA-3, Qwen2.5-1.5B) using 4-bit Quantized Low-Rank Adaptation (QLoRA) on curated oncology instruction pairs.", bullet_style))
    story.append(Paragraph("• <b>Multi-Axis Evaluation & Safety Firewall:</b> Subjected models to adversarial clinical prompts, out-of-distribution progress notes, and established a <b>6-gate safety firewall</b> (tau* calibration) to block unsafe recommendations.", bullet_style))
    story.append(Paragraph("• <b>100% Offline CPU Integration:</b> Exported models to GGUF (Q4_K_M) running locally on CPU via <code>llama.cpp</code> and FastAPI (<code>/summarize</code>), accompanied by a clinician desktop web interface.", bullet_style))

    # Stage 5
    story.append(Spacer(1, 4))
    story.append(Paragraph("Stage 5: GenAI Synthetic Oncology Stress-Test Engine (Master Milestone)", h1_style))
    story.append(Paragraph(
        "<b>Objective:</b> Create an autonomous adversarial engine that generates statistically valid, biologically grounded, "
        "and clinically realistic synthetic oncology patient cases to stress-test and discover edge-case vulnerabilities in Stages 1–4.",
        body_style
    ))
    story.append(Paragraph(
        "Stage 5 was structured across <b>5 professional engineering roles</b>:",
        body_style
    ))

    st5_roles_data = [
        [Paragraph("Role", table_hdr), Paragraph("Scope & Core Technical Responsibilities", table_hdr), Paragraph("Status", table_hdr)],
        [
            Paragraph("<b>Data Engineer</b>", table_cell_bold),
            Paragraph("Constructed empirical reference distributions, continuous bounds, biological constraint matrices, and 16 curated evidence chunks (NCCN, CTCAE v5.0, FDA bulletins).", table_cell),
            Paragraph("<font color='#059669'><b>VERIFIED (100%)</b></font>", table_cell)
        ],
        [
            Paragraph("<b>EDA / Prompt Engineer</b>", table_cell_bold),
            Paragraph("Mined empirical failure modes (FP-001–FP-006), cataloged and ranked 15 blind spots (BS01–BS15), authored prompt library (PROMPT-R01 to R15), and formulated drift immutability rules.", table_cell),
            Paragraph("<font color='#059669'><b>VERIFIED (100%)</b></font>", table_cell)
        ],
        [
            Paragraph("<b>Gen AI Engineer</b>", table_cell_bold),
            Paragraph("Built two-stage hybrid generation: Monte Carlo <code>StructuredSampler</code> + NVIDIA LLaMA-3.2-11B/3.1-70B <code>NarrativeGenerator</code>, TF-IDF RAG retriever, and counterfactual sensitivity checks.", table_cell),
            Paragraph("<font color='#059669'><b>VERIFIED (100%)</b></font>", table_cell)
        ],
        [
            Paragraph("<b>Evaluation Engineer</b>", table_cell_bold),
            Paragraph("Constructed 10-step evaluation harness, F01–F20 failure taxonomy, system difficulty scoring, candidate gatekeeper, and 8 comprehensive technical markdown reports.", table_cell),
            Paragraph("<font color='#059669'><b>VERIFIED (100%)</b></font>", table_cell)
        ],
        [
            Paragraph("<b>Integration Engineer</b>", table_cell_bold),
            Paragraph("Built unified root CLI (<code>run_stage5.py</code>), SQLite WAL result repository (<code>stage5.db</code>), downstream multi-stage adapters, wildcard ranker, and FastAPI dashboard backend.", table_cell),
            Paragraph("<font color='#059669'><b>VERIFIED (100%)</b></font>", table_cell)
        ]
    ]
    t_st5 = Table(st5_roles_data, colWidths=[95, 335, 89])
    t_st5.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), bg_card),
        ('BOX', (0, 0), (-1, -1), 0.7, border_color),
        ('INNERGRID', (0, 0), (-1, -1), 0.4, border_color),
        ('TOPPADDING', (0, 0), (-1, -1), 3),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
        ('LEFTPADDING', (0, 0), (-1, -1), 4.5),
        ('RIGHTPADDING', (0, 0), (-1, -1), 4.5),
    ]))
    story.append(t_st5)
    story.append(Spacer(1, 5))

    story.append(Paragraph("Core Technical Innovations in Stage 5:", h2_style))
    story.append(Paragraph("• <b>Hybrid Generation Architecture:</b> Statistical Monte Carlo sampling ensures 100% biological/pharmacological validity (no hallucinated drug dosages or impossible mutations), while NVIDIA LLaMA-3.2/3.1 creates authentic physician SOAP notes.", bullet_style))
    story.append(Paragraph("• <b>RAG Grounding & Zero-Hallucination Guardrails:</b> TF-IDF vector index retrieves approved excerpts from NCCN NSCLC Guidelines v4.2024 and CTCAE v5.0, paired with an automated narrative validator that checks LLM output against the patient's ground truth.", bullet_style))
    story.append(Paragraph("• <b>Multi-Stage Downstream Stress Testing:</b> Generated profiles are actively fed through Stages 1, 2, 3, and 4 to uncover blind spots (e.g. Stage 1 tabular model predicting low risk despite metastatic ctDNA spikes; Stage 4 SLM dropping MET bypass resistance).", bullet_style))
    story.append(Paragraph("• <b>Interactive Localhost Web Platform (Port 8085):</b> Custom-built interactive UI featuring seed condition selectors, real-time patient synthesis, multi-bar validation scorecards (Statistical 92%, Biological 90%, Clinical 87%), and dynamic scenario distribution donut charts.", bullet_style))

    # ==================== PAGE 3 ====================
    story.append(PageBreak())

    # --- Section 3: Web Applications & Dashboards ---
    story.append(Paragraph("3. Interactive Web Dashboards & UI Showcases", h1_style))
    story.append(Paragraph(
        "The project provides two interactive, user-facing web applications for clinical demonstration and system monitoring:",
        body_style
    ))
    story.append(Paragraph("1. <b>Synthetic Patient Generator Dashboard (Stage 5):</b> Hosted locally on <b>http://127.0.0.1:8085</b>. Implements a responsive medical interface with live patient generation, clinical narrative viewer, downstream stress-test harness, scenario history table with CSV export, and analytics charts.", bullet_style))
    story.append(Paragraph("2. <b>ONCO.AI 3D Showcase (<code>onco-showcase/</code>):</b> A modern web application built with React 19, Three.js, and GSAP ScrollTrigger. Delivers a 3D animated walkthrough of the proposed six-stage oncology intelligence workflow with accessible evidence dialogs.", bullet_style))

    # --- Section 4: Testing & Governance ---
    story.append(Paragraph("4. Verification, Testing & Codebase Governance", h1_style))
    
    metrics_data = [
        [Paragraph("Metric / Audit Criterion", table_hdr), Paragraph("Achieved Value", table_hdr), Paragraph("Verification Mechanism", table_hdr)],
        [Paragraph("Automated Unit Test Suites", table_cell_bold), Paragraph("<b>116+ Tests Passed (100%)</b>", table_cell), Paragraph("PyTest across Data, EDA, GenAI, Eval & Integration modules", table_cell)],
        [Paragraph("Programmatic Definition-of-Done", table_cell_bold), Paragraph("<b>100% Passed (Q1–Q10)</b>", table_cell), Paragraph("Independent <code>verify_done.py</code> audits in all stages", table_cell)],
        [Paragraph("Toxicity Prediction Macro F1", table_cell_bold), Paragraph("<b>0.7557</b> (Stage 3 Baselines)", table_cell), Paragraph("Locked test set with 95% bootstrap confidence intervals", table_cell)],
        [Paragraph("Patient Synthesis Realism", table_cell_bold), Paragraph("<b>92.4% Statistical, 90.1% Bio</b>", table_cell), Paragraph("Stage 5 Multi-axis constraint and drift validator", table_cell)],
        [Paragraph("Local Storage & Audit Lineage", table_cell_bold), Paragraph("<b>100% Deterministic Provenance</b>", table_cell), Paragraph("SQLite WAL mode (<code>stage5.db</code>) & JSON manifests", table_cell)],
        [Paragraph("Private Offline Deployment", table_cell_bold), Paragraph("<b>100% Offline CPU Capable</b>", table_cell), Paragraph("GGUF Q4_K_M + llama.cpp local inference service", table_cell)]
    ]
    t_metrics = Table(metrics_data, colWidths=[170, 135, 214])
    t_metrics.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), bg_card),
        ('BOX', (0, 0), (-1, -1), 0.7, border_color),
        ('INNERGRID', (0, 0), (-1, -1), 0.4, border_color),
        ('TOPPADDING', (0, 0), (-1, -1), 3),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
        ('LEFTPADDING', (0, 0), (-1, -1), 4.5),
        ('RIGHTPADDING', (0, 0), (-1, -1), 4.5),
    ]))
    story.append(t_metrics)
    story.append(Spacer(1, 6))

    # --- Section 5: Viva, Interview & Academic Readiness ---
    story.append(Paragraph("5. Viva, Interview & Academic Readiness", h1_style))
    story.append(Paragraph(
        "The repository contains exhaustive study materials, viva defense guides, and interview transcripts in <code>docs/</code>:",
        body_style
    ))
    story.append(Paragraph("• <b>Level 1 Viva Preparation Guide:</b> <code>docs/viva-preparation/Data_Science_Viva_Level_1_Study_Material.pdf</code> covering core ML theory, statistical distributions, regularization, metrics, and cross-validation.", bullet_style))
    story.append(Paragraph("• <b>Stage 3 & 4 Master Viva Guide:</b> Detailed HTML and Markdown guides breaking down Clinical NLP tokenization, BioLinkBERT attention mechanisms, LoRA mathematical rank factorization, and tau* calibration.", bullet_style))
    story.append(Paragraph("• <b>Project Interview Preparation:</b> <code>docs/PROJECT_INTERVIEW_PREPARATION.md</code> detailing senior engineering answers, trade-off justifications, failure modes, and architectural decisions.", bullet_style))

    # --- Section 6: Key Repository Commands & Quickstart ---
    story.append(Paragraph("6. Key Operational Commands & Quickstart", h1_style))
    
    cmd_data = [
        [Paragraph("Operational Action", table_hdr), Paragraph("Command Line Execution", table_hdr), Paragraph("Output & Target Service", table_hdr)],
        [
            Paragraph("<b>Run Stage 5 Generator</b>", table_cell_bold),
            Paragraph("<font color='#0F766E'>python run_stage5.py --n 20 --seed 42</font>", code_style),
            Paragraph("Generates 20 scenarios, runs stress tests, ranks wildcards into SQLite", table_cell)
        ],
        [
            Paragraph("<b>Launch Stage 5 Web UI</b>", table_cell_bold),
            Paragraph("<font color='#0F766E'>python run_dashboard.py</font>", code_style),
            Paragraph("Starts FastAPI dashboard server at <b>http://127.0.0.1:8085</b>", table_cell)
        ],
        [
            Paragraph("<b>Stage 1 Inference API</b>", table_cell_bold),
            Paragraph("<font color='#0F766E'>uvicorn src.app:app --port 8000</font>", code_style),
            Paragraph("Launches Tabular ML toxicity prediction REST endpoints (/predict)", table_cell)
        ],
        [
            Paragraph("<b>Offline SLM Inference</b>", table_cell_bold),
            Paragraph("<font color='#0F766E'>uvicorn src.app:app --port 8080</font>", code_style),
            Paragraph("Stage 4 GGUF local CPU reasoning service (/summarize)", table_cell)
        ],
        [
            Paragraph("<b>Run DoD Verification</b>", table_cell_bold),
            Paragraph("<font color='#0F766E'>python \"stage 5 Gen-AI/verify_done.py\"</font>", code_style),
            Paragraph("Runs automated Definition-of-Done programmatic audits", table_cell)
        ]
    ]
    t_cmd = Table(cmd_data, colWidths=[120, 205, 194])
    t_cmd.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), bg_card),
        ('BOX', (0, 0), (-1, -1), 0.7, border_color),
        ('INNERGRID', (0, 0), (-1, -1), 0.4, border_color),
        ('TOPPADDING', (0, 0), (-1, -1), 3),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
        ('LEFTPADDING', (0, 0), (-1, -1), 4.5),
        ('RIGHTPADDING', (0, 0), (-1, -1), 4.5),
    ]))
    story.append(t_cmd)
    story.append(Spacer(1, 6))

    # --- Section 7: Future Work & Next Steps ---
    story.append(Paragraph("7. Clinical Impact & Prospective Roadmap", h1_style))
    story.append(Paragraph(
        "By uniting statistical rigor, deep multimodal embeddings, clinical NLP, localized SLMs, and an adversarial GenAI stress-testing "
        "harness, this repository establishes a comprehensive benchmark for AI safety in precision oncology. "
        "Immediate roadmap priorities include prospective multi-center cohort validation, direct hospital EHR FHIR integration, "
        "and expanding the RAG vector index to encompass international clinical trial protocols (ASCO, ESMO).",
        body_style
    ))

    # Build the document
    doc.build(story, canvasmaker=NumberedCanvas)
    print(f"[OK] Master Summary PDF built successfully: {out_path}")


if __name__ == "__main__":
    out_dir = Path("output/pdf")
    out_dir.mkdir(parents=True, exist_ok=True)
    target_pdf = out_dir / "Oncology_Treatment_Project_Comprehensive_Summary.pdf"
    build_pdf(str(target_pdf))
    
    # Also copy to root for quick access
    root_pdf = Path("Oncology_Treatment_Project_Summary.pdf")
    import shutil
    shutil.copyfile(target_pdf, root_pdf)
    print(f"[OK] Root copy created: {root_pdf}")
