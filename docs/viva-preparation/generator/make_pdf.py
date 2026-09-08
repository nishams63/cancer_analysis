"""
Master Script: Assembles all modules into a unified HTML document and converts it to PDF via Microsoft Edge.
"""
import os
import sys
import subprocess
import html

from css_styles import CSS_CONTENT
from part1_ml import ML_CONCEPTS
from part2_dl import DL_CONCEPTS
from part3_roles import ROLES_DATA
from part4_tables import COMPARISON_TABLES
from part5_questions import (
    VIVA_BASIC_QUESTIONS,
    VIVA_INTERMEDIATE_QUESTIONS,
    VIVA_WHY_QUESTIONS,
    VIVA_PRACTICAL_QUESTIONS,
    VIVA_RAPID_FIRE_QUESTIONS
)
from part6_revision import (
    REVISION_ML_POINTS,
    REVISION_DL_POINTS,
    REVISION_EDA_POINTS,
    REVISION_ROLE_POINTS,
    TOP_25_MUST_KNOW_QUESTIONS
)

def build_html():
    out = []
    out.append("<!DOCTYPE html>")
    out.append("<html lang='en'>")
    out.append("<head>")
    out.append("  <meta charset='UTF-8'>")
    out.append("  <meta name='viewport' content='width=device-width, initial-scale=1.0'>")
    out.append("  <title>DATA SCIENCE VIVA PREPARATION — Level 1</title>")
    out.append(f"  <style>{CSS_CONTENT}</style>")
    out.append("</head>")
    out.append("<body>")
    out.append("<div class='container'>")

    # Cover & Header
    out.append("<header class='cover-header'>")
    out.append("  <div class='badge badge-dark'>B.Tech Artificial Intelligence &amp; Data Science</div>")
    out.append("  <h1 style='margin-top:10px;'>DATA SCIENCE VIVA PREPARATION</h1>")
    out.append("  <div style='font-size:18px; font-weight:700; color:#cbd5e1; margin-bottom:6px;'>Level 1 — Machine Learning, Deep Learning &amp; Engineering Roles</div>")
    out.append("  <div class='subtitle'>Easy Concepts + Viva Questions + Quick Revision</div>")
    out.append("  <div class='project-badge-bar'>")
    out.append("    <span class='badge badge-green'>Project-Grounded Study Material</span>")
    out.append("    <span class='badge badge-blue'>Stage 1 ML: Toxicity Risk Prediction</span>")
    out.append("    <span class='badge badge-amber'>Stage 2 DL: Multimodal Progression</span>")
    out.append("    <span class='badge badge-dark'>FastAPI + Locked Test Set Benchmark</span>")
    out.append("  </div>")
    out.append("</header>")

    # Project Context Box
    out.append("<div class='project-context-box'>")
    out.append("  <strong>Project Context Grounding:</strong> This viva study material is explicitly contextualized to our project: ")
    out.append("  <em>'Personalized Precision Medicine for Oncology Treatment Optimization'</em>. ")
    out.append("  Every definition, comparison, and viva answer connects directly to our real-world system: ")
    out.append("  <strong>Stage 1 ML</strong> (predicting <code>toxicity_risk</code> [Low/Moderate/High] using LightGBM Candidate V4 on 1,750 encounters) and ")
    out.append("  <strong>Stage 2 DL</strong> (histopathology CNN tile classification + longitudinal ctDNA LSTM progression forecasting + Multimodal Late Fusion FastAPI service).")
    out.append("</div>")

    # Table of Contents Overview
    out.append("<div class='concept-card' style='background:#f8fafc; border-left:4px solid #1e3a8a; margin-bottom:24px;'>")
    out.append("  <div style='font-weight:700; font-size:14px; color:#1e3a8a; margin-bottom:8px;'>STUDY GUIDE STRUCTURE:</div>")
    out.append("  <ul style='margin-left:20px; font-size:12.5px; color:#334155; line-height:1.7;'>")
    out.append("    <li><strong>Part 1 — Machine Learning:</strong> 24 Core Concepts (Definitions, Project Examples, Viva Answers, Key Takeaways).</li>")
    out.append("    <li><strong>Part 2 — Deep Learning:</strong> 27 Core Concepts (How it works, Neural architectures, CNNs, LSTMs, Transformers).</li>")
    out.append("    <li><strong>Part 3 — Data Science Engineering Roles:</strong> 5 Industry Roles (Data Eng, EDA Eng, ML Eng, Eval Eng, Integration Eng).</li>")
    out.append("    <li><strong>Part 4 — Important Differences:</strong> 13 Essential Comparison Tables for viva clarity.</li>")
    out.append("    <li><strong>Part 5 — Viva Questions:</strong> 95 Curated Questions (Basic, Intermediate, 'Why' Questions, Practical Scenarios, 25 Rapid-Fire).</li>")
    out.append("    <li><strong>Final Section — '1 Hour Before Viva':</strong> Rapid Revision Points + Top 25 Must-Know Questions.</li>")
    out.append("  </ul>")
    out.append("</div>")

    out.append("<div class='page-break'></div>")

    # ================= PART 1: MACHINE LEARNING =================
    out.append("<div class='part-title-banner'>")
    out.append("  <h2>PART 1 — MACHINE LEARNING</h2>")
    out.append("  <span class='badge badge-dark'>24 Core Concepts</span>")
    out.append("</div>")
    out.append("<p class='section-intro'>Format: Concept &rarr; Simple Definition &rarr; Project Example &rarr; Viva-Ready Answer &rarr; Key Points to Remember.</p>")

    for c in ML_CONCEPTS:
        out.append("<div class='concept-card highlight'>")
        out.append("  <div class='concept-header'>")
        out.append(f"    <div class='concept-title'>{c['title']}</div>")
        out.append(f"    <div class='concept-num'>ML Concept #{c['num']}</div>")
        out.append("  </div>")
        out.append(f"  <div class='def-block'><strong>Simple Definition:</strong> {c['def']}</div>")
        out.append(f"  <div class='example-box'><strong>Project Example:</strong> {c['example']}</div>")
        out.append(f"  <div class='viva-box'><strong>Viva Answer (Say to Examiner):</strong> \"{c['viva']}\"</div>")
        out.append(f"  <div class='remember-box'><strong>Remember:</strong> {c['remember']}</div>")
        out.append("</div>")

    out.append("<div class='page-break'></div>")

    # ================= PART 2: DEEP LEARNING =================
    out.append("<div class='part-title-banner'>")
    out.append("  <h2>PART 2 — DEEP LEARNING</h2>")
    out.append("  <span class='badge badge-dark'>27 Core Concepts</span>")
    out.append("</div>")
    out.append("<p class='section-intro'>Format: Simple Definition &rarr; How It Works &rarr; Small Project Example &rarr; Viva-Ready Answer &rarr; Key Point to Remember.</p>")

    for c in DL_CONCEPTS:
        out.append("<div class='concept-card' style='border-left: 4px solid #a855f7;'>")
        out.append("  <div class='concept-header'>")
        out.append(f"    <div class='concept-title' style='color:#581c87;'>{c['title']}</div>")
        out.append(f"    <div class='concept-num' style='background:#f3e8ff; color:#7e22ce;'>DL Concept #{c['num']}</div>")
        out.append("  </div>")
        out.append(f"  <div class='def-block'><strong>Simple Definition:</strong> {c['def']}</div>")
        out.append(f"  <div class='how-block'><strong>How It Works:</strong> {c['how']}</div>")
        out.append(f"  <div class='example-box'><strong>Project Example:</strong> {c['example']}</div>")
        out.append(f"  <div class='viva-box'><strong>Viva-Ready Answer:</strong> \"{c['viva']}\"</div>")
        out.append(f"  <div class='remember-box'><strong>Key Point to Remember:</strong> {c['remember']}</div>")
        out.append("</div>")

    out.append("<div class='page-break'></div>")

    # ================= PART 3: ROLES =================
    out.append("<div class='part-title-banner'>")
    out.append("  <h2>PART 3 — DATA SCIENCE ENGINEERING ROLES</h2>")
    out.append("  <span class='badge badge-dark'>5 Industry Roles</span>")
    out.append("</div>")
    out.append("<p class='section-intro'>A practical breakdown of daily duties, technical pipelines, tools, and viva questions for each role in our oncology project.</p>")

    for r in ROLES_DATA:
        out.append("<div class='role-card'>")
        out.append("  <div class='role-title'>")
        out.append(f"    <span>{r['role']}</span>")
        out.append(f"    <span class='role-tag'>{r['tag']}</span>")
        out.append("  </div>")
        out.append(f"  <p style='font-size:13px; color:#334155; margin-bottom:12px;'><strong>Role Summary:</strong> {r['summary']}</p>")
        
        out.append("  <div class='role-section'>")
        out.append(f"    <h4>What They Do:</h4><p>{r['what_they_do']}</p>")
        
        if "etl_elt" in r:
            out.append(f"    <h4>Data Collection &amp; ETL/ELT:</h4><p>{r['data_collection']}<br>{r['etl_elt']}</p>")
            out.append(f"    <h4>Databases &amp; Cleaning:</h4><p>{r['databases']}<br>{r['data_cleaning_storage']}</p>")
        
        if "components" in r:
            out.append(f"    <h4>Why EDA &amp; Core Analytical Pillars:</h4><p>{r['why_eda']}<br>{r['components']}</p>")
        
        if "core_tasks" in r:
            out.append(f"    <h4>Core Responsibilities:</h4><p>{r['core_tasks']}</p>")
        
        if "ds_vs_mle" in r:
            out.append(f"    <h4>Data Scientist vs ML Engineer:</h4><p>{r['ds_vs_mle']}</p>")
            
        out.append(f"    <h4>Common Tools:</h4><div class='role-tools'>{r['tools']}</div>")
        out.append("  </div>")
        
        out.append(f"  <div class='example-box' style='margin-top:12px;'><strong>Project Application:</strong> {r['project_example']}</div>")
        out.append(f"  <div class='viva-box'><strong>Viva Question:</strong> \"{r['viva_q1']}\"<br><strong>Viva Answer:</strong> {r['viva_a1']}</div>")
        out.append(f"  <div class='viva-box' style='margin-top:8px;'><strong>Viva Question:</strong> \"{r['viva_q2']}\"<br><strong>Viva Answer:</strong> {r['viva_a2']}</div>")
        out.append("</div>")

    out.append("<div class='page-break'></div>")

    # ================= PART 4: COMPARISON TABLES =================
    out.append("<div class='part-title-banner'>")
    out.append("  <h2>PART 4 — IMPORTANT DIFFERENCES</h2>")
    out.append("  <span class='badge badge-dark'>13 Comparison Tables</span>")
    out.append("</div>")
    out.append("<p class='section-intro'>Structured comparison tables directly contrasting key concepts and roles with project-based context.</p>")

    for t in COMPARISON_TABLES:
        out.append(f"<h3 style='font-size:15px; font-weight:700; color:#1e293b; margin:16px 0 8px 0;'>Table {t['num']}: {t['title']}</h3>")
        out.append("<div class='table-container'>")
        out.append("  <table>")
        out.append("    <thead><tr>")
        for h in t['headers']:
            out.append(f"      <th>{h}</th>")
        out.append("    </tr></thead>")
        out.append("    <tbody>")
        for row in t['rows']:
            out.append("    <tr>")
            for i, cell in enumerate(row):
                if i == 0:
                    out.append(f"      <td style='font-weight:700;'>{cell}</td>")
                else:
                    out.append(f"      <td>{cell}</td>")
            out.append("    </tr>")
        out.append("    </tbody>")
        out.append("  </table>")
        out.append("</div>")

    out.append("<div class='page-break'></div>")

    # ================= PART 5: VIVA QUESTIONS =================
    out.append("<div class='part-title-banner'>")
    out.append("  <h2>PART 5 — VIVA QUESTIONS &amp; ANSWERS</h2>")
    out.append("  <span class='badge badge-dark'>95 Questions with Model Answers</span>")
    out.append("</div>")
    out.append("<p class='section-intro'>Curated interview questions divided into Basic, Intermediate, 'Why' Questions, Practical Scenarios, and Rapid-Fire revision.</p>")

    # Section 1: Basic
    out.append("<h3 style='font-size:15px; font-weight:800; color:#1e3a8a; margin:18px 0 10px 0; border-bottom:2px solid #93c5fd; padding-bottom:4px;'>SECTION A: BASIC QUESTIONS (Questions 1 to 18)</h3>")
    for i, item in enumerate(VIVA_BASIC_QUESTIONS, 1):
        out.append("<div class='qa-card'>")
        out.append(f"  <div class='qa-q'><span class='q-num'>Q{i}</span>{item['q']}</div>")
        out.append(f"  <div class='qa-a'>{item['a']}</div>")
        out.append("</div>")

    # Section 2: Intermediate
    out.append("<div class='page-break'></div>")
    out.append("<h3 style='font-size:15px; font-weight:800; color:#1e3a8a; margin:18px 0 10px 0; border-bottom:2px solid #93c5fd; padding-bottom:4px;'>SECTION B: INTERMEDIATE QUESTIONS (Questions 19 to 36)</h3>")
    for i, item in enumerate(VIVA_INTERMEDIATE_QUESTIONS, 19):
        out.append("<div class='qa-card'>")
        out.append(f"  <div class='qa-q'><span class='q-num'>Q{i}</span>{item['q']}</div>")
        out.append(f"  <div class='qa-a'>{item['a']}</div>")
        out.append("</div>")

    # Section 3: Why
    out.append("<div class='page-break'></div>")
    out.append("<h3 style='font-size:15px; font-weight:800; color:#1e3a8a; margin:18px 0 10px 0; border-bottom:2px solid #93c5fd; padding-bottom:4px;'>SECTION C: 'WHY' QUESTIONS (Questions 37 to 54)</h3>")
    for i, item in enumerate(VIVA_WHY_QUESTIONS, 37):
        out.append("<div class='qa-card'>")
        out.append(f"  <div class='qa-q'><span class='q-num' style='background:#a855f7;'>Q{i}</span>{item['q']}</div>")
        out.append(f"  <div class='qa-a'>{item['a']}</div>")
        out.append("</div>")

    # Section 4: Practical
    out.append("<div class='page-break'></div>")
    out.append("<h3 style='font-size:15px; font-weight:800; color:#1e3a8a; margin:18px 0 10px 0; border-bottom:2px solid #93c5fd; padding-bottom:4px;'>SECTION D: PRACTICAL &amp; SCENARIO QUESTIONS (Questions 55 to 70)</h3>")
    for i, item in enumerate(VIVA_PRACTICAL_QUESTIONS, 55):
        out.append("<div class='qa-card'>")
        out.append(f"  <div class='qa-q'><span class='q-num' style='background:#059669;'>Q{i}</span>{item['q']}</div>")
        out.append(f"  <div class='qa-a'>{item['a']}</div>")
        out.append("</div>")

    # Section 5: Rapid Fire
    out.append("<div class='page-break'></div>")
    out.append("<h3 style='font-size:15px; font-weight:800; color:#1e3a8a; margin:18px 0 10px 0; border-bottom:2px solid #93c5fd; padding-bottom:4px;'>SECTION E: RAPID-FIRE QUESTIONS (25 One-Line Q&amp;As for Quick Revision)</h3>")
    out.append("<div class='rapid-grid'>")
    for item in VIVA_RAPID_FIRE_QUESTIONS:
        out.append("<div class='rapid-item'>")
        out.append(f"  <div class='rapid-q'>{item['q']}</div>")
        out.append(f"  <div class='rapid-a'>{item['a']}</div>")
        out.append("</div>")
    out.append("</div>")

    out.append("<div class='page-break'></div>")

    # ================= FINAL SECTION: 1 HOUR BEFORE VIVA =================
    out.append("<div class='part-title-banner' style='background:linear-gradient(90deg, #b45309 0%, #d97706 100%);'>")
    out.append("  <h2>FINAL SECTION — '1 HOUR BEFORE VIVA' QUICK-REVISION SHEET</h2>")
    out.append("  <span class='badge badge-dark'>High-Yield Summary</span>")
    out.append("</div>")
    out.append("<p class='section-intro'>Review these high-yield bullet checklists right before entering your viva examination.</p>")

    # 30 ML points
    out.append("<div class='rev-box'>")
    out.append("  <h3>30 Most Important Machine Learning Points</h3>")
    out.append("  <ul class='rev-list'>")
    for p in REVISION_ML_POINTS:
        out.append(f"    <li>{p}</li>")
    out.append("  </ul>")
    out.append("</div>")

    # 30 DL points
    out.append("<div class='page-break'></div>")
    out.append("<div class='rev-box'>")
    out.append("  <h3>30 Most Important Deep Learning Points</h3>")
    out.append("  <ul class='rev-list'>")
    for p in REVISION_DL_POINTS:
        out.append(f"    <li>{p}</li>")
    out.append("  </ul>")
    out.append("</div>")

    # 15 EDA points
    out.append("<div class='rev-box'>")
    out.append("  <h3>15 Important Exploratory Data Analysis (EDA) Points</h3>")
    out.append("  <ul class='rev-list'>")
    for p in REVISION_EDA_POINTS:
        out.append(f"    <li>{p}</li>")
    out.append("  </ul>")
    out.append("</div>")

    # Engineering Roles Points
    out.append("<div class='page-break'></div>")
    out.append("<div class='rev-box'>")
    out.append("  <h3>10 Data Engineer Points</h3>")
    out.append("  <ul class='rev-list'>")
    for p in REVISION_ROLE_POINTS["de"]:
        out.append(f"    <li>{p}</li>")
    out.append("  </ul>")
    out.append("</div>")

    out.append("<div class='rev-box'>")
    out.append("  <h3>10 ML Engineer Points</h3>")
    out.append("  <ul class='rev-list'>")
    for p in REVISION_ROLE_POINTS["mle"]:
        out.append(f"    <li>{p}</li>")
    out.append("  </ul>")
    out.append("</div>")

    out.append("<div class='rev-box'>")
    out.append("  <h3>10 Evaluation Engineer Points</h3>")
    out.append("  <ul class='rev-list'>")
    for p in REVISION_ROLE_POINTS["eval"]:
        out.append(f"    <li>{p}</li>")
    out.append("  </ul>")
    out.append("</div>")

    out.append("<div class='rev-box'>")
    out.append("  <h3>10 Integration Engineer Points</h3>")
    out.append("  <ul class='rev-list'>")
    for p in REVISION_ROLE_POINTS["integ"]:
        out.append(f"    <li>{p}</li>")
    out.append("  </ul>")
    out.append("</div>")

    # Top 25 Must-Know Questions
    out.append("<div class='page-break'></div>")
    out.append("<div class='rev-box' style='border:2px solid #2563eb;'>")
    out.append("  <h3 style='color:#1d4ed8; font-size:16px;'>TOP 25 QUESTIONS THE STUDENT MUST KNOW BEFORE THE VIVA</h3>")
    out.append("  <div style='margin-top:12px;'>")
    for q, a in TOP_25_MUST_KNOW_QUESTIONS:
        out.append("    <div style='margin-bottom:12px; padding:8px 12px; background:#eff6ff; border-radius:6px; border-left:3px solid #3b82f6;'>")
        out.append(f"      <div style='font-weight:700; color:#1e3a8a; font-size:12.5px;'>{q}</div>")
        out.append(f"      <div style='color:#1e293b; font-size:12px; margin-top:3px;'>{a}</div>")
        out.append("    </div>")
    out.append("  </div>")
    out.append("</div>")

    # Footer note
    out.append("<footer style='text-align:center; padding:20px; font-size:11.5px; color:#64748b; border-top:1px solid #e2e8f0; margin-top:30px;'>")
    out.append("  <strong>DATA SCIENCE VIVA STUDY MATERIAL — LEVEL 1</strong><br>")
    out.append("  Department of Artificial Intelligence &amp; Data Science | Grounded in Personalized Precision Oncology Treatment Optimization System")
    out.append("</footer>")

    out.append("</div>") # container
    out.append("</body>")
    out.append("</html>")
    return "\n".join(out)

def main():
    print("Building HTML study material...")
    html_content = build_html()
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.dirname(script_dir)  # docs/viva-preparation
    html_path = os.path.join(output_dir, "Data_Science_Viva_Level_1_Study_Material.html")
    pdf_path = os.path.join(output_dir, "Data_Science_Viva_Level_1_Study_Material.pdf")
    
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html_content)
    print(f"HTML generated at: {html_path} ({len(html_content)} bytes)")
    
    edge_path = r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"
    if not os.path.exists(edge_path):
        edge_path = r"C:\Program Files\Microsoft\Edge\Application\msedge.exe"
    
    print(f"Using Edge at: {edge_path}")
    print("Converting HTML to high-quality PDF via Microsoft Edge headless...")
    
    cmd = [
        edge_path,
        "--headless=new",
        "--disable-gpu",
        "--no-pdf-header-footer",
        f"--print-to-pdf={pdf_path}",
        html_path
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    if os.path.exists(pdf_path):
        size_kb = os.path.getsize(pdf_path) / 1024
        print(f"SUCCESS! PDF successfully created at:\n{pdf_path}\nFile size: {size_kb:.1f} KB")
    else:
        print("ERROR: PDF was not generated. Stderr:", result.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
