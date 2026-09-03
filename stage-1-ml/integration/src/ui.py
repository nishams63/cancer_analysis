"""
Web Interface for Stage 1 ML Toxicity Risk Prediction Service.
Role: Integration Engineer.

Provides a modern, responsive web application for oncology clinical researchers
to evaluate patient toxicity risk using the frozen Candidate V4 model.
"""

def get_ui_html() -> str:
    return """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>OncoRisk AI — Precision Oncology Toxicity Prediction</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Outfit:wght@500;600;700&display=swap" rel="stylesheet">
  <style>
    :root {
      --bg-base: #080c14;
      --bg-surface: #0f172a;
      --bg-card: rgba(30, 41, 59, 0.65);
      --border-card: rgba(255, 255, 255, 0.08);
      --primary: #38bdf8;
      --primary-gradient: linear-gradient(135deg, #0ea5e9, #6366f1);
      --text-main: #f1f5f9;
      --text-muted: #94a3b8;
      --risk-low: #10b981;
      --risk-low-bg: rgba(16, 185, 129, 0.15);
      --risk-mod: #f59e0b;
      --risk-mod-bg: rgba(245, 158, 11, 0.15);
      --risk-high: #ef4444;
      --risk-high-bg: rgba(239, 68, 68, 0.15);
    }
    * {
      box-sizing: border-box;
      margin: 0;
      padding: 0;
    }
    body {
      font-family: 'Inter', sans-serif;
      background-color: var(--bg-base);
      color: var(--text-main);
      line-height: 1.5;
      padding: 24px;
      min-height: 100vh;
    }
    .container {
      max-width: 1280px;
      margin: 0 auto;
    }
    header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      padding-bottom: 24px;
      border-bottom: 1px solid var(--border-card);
      margin-bottom: 24px;
    }
    .brand-title {
      font-family: 'Outfit', sans-serif;
      font-size: 26px;
      font-weight: 700;
      background: var(--primary-gradient);
      -webkit-background-clip: text;
      -webkit-text-fill-color: transparent;
      display: flex;
      align-items: center;
      gap: 10px;
    }
    .brand-subtitle {
      font-size: 13px;
      color: var(--text-muted);
      margin-top: 4px;
    }
    .header-pills {
      display: flex;
      align-items: center;
      gap: 12px;
    }
    .badge {
      display: inline-flex;
      align-items: center;
      gap: 6px;
      padding: 6px 12px;
      border-radius: 9999px;
      font-size: 12px;
      font-weight: 600;
      border: 1px solid var(--border-card);
      background: rgba(15, 23, 42, 0.8);
    }
    .badge-dot {
      width: 8px;
      height: 8px;
      border-radius: 50%;
      background: #10b981;
      box-shadow: 0 0 8px #10b981;
    }
    .docs-link {
      text-decoration: none;
      color: var(--primary);
      font-size: 13px;
      font-weight: 600;
      transition: color 0.2s;
    }
    .docs-link:hover {
      color: #93c5fd;
    }
    .preset-bar {
      display: flex;
      align-items: center;
      gap: 10px;
      background: var(--bg-surface);
      border: 1px solid var(--border-card);
      border-radius: 12px;
      padding: 12px 16px;
      margin-bottom: 24px;
      flex-wrap: wrap;
    }
    .preset-label {
      font-size: 13px;
      font-weight: 600;
      color: var(--text-muted);
      margin-right: 4px;
    }
    .btn-preset {
      padding: 6px 14px;
      border-radius: 8px;
      border: 1px solid var(--border-card);
      background: rgba(255, 255, 255, 0.05);
      color: var(--text-main);
      font-size: 12px;
      font-weight: 500;
      cursor: pointer;
      transition: all 0.2s;
    }
    .btn-preset:hover {
      background: rgba(255, 255, 255, 0.12);
      border-color: rgba(255, 255, 255, 0.2);
    }
    .btn-preset.low:hover {
      color: var(--risk-low);
      border-color: var(--risk-low);
    }
    .btn-preset.mod:hover {
      color: var(--risk-mod);
      border-color: var(--risk-mod);
    }
    .btn-preset.high:hover {
      color: var(--risk-high);
      border-color: var(--risk-high);
    }
    .grid-cards {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
      gap: 20px;
      margin-bottom: 24px;
    }
    .card {
      background: var(--bg-card);
      backdrop-filter: blur(12px);
      border: 1px solid var(--border-card);
      border-radius: 16px;
      padding: 20px;
      display: flex;
      flex-direction: column;
      gap: 14px;
    }
    .card-title {
      font-family: 'Outfit', sans-serif;
      font-size: 15px;
      font-weight: 600;
      color: var(--primary);
      display: flex;
      align-items: center;
      gap: 8px;
      padding-bottom: 8px;
      border-bottom: 1px solid rgba(255, 255, 255, 0.05);
    }
    .form-group {
      display: flex;
      flex-direction: column;
      gap: 4px;
    }
    label {
      font-size: 11px;
      font-weight: 600;
      color: var(--text-muted);
      text-transform: uppercase;
      letter-spacing: 0.5px;
    }
    input, select {
      background: rgba(15, 23, 42, 0.8);
      border: 1px solid rgba(255, 255, 255, 0.1);
      border-radius: 8px;
      padding: 8px 12px;
      color: var(--text-main);
      font-size: 13px;
      outline: none;
      transition: all 0.2s;
    }
    input:focus, select:focus {
      border-color: var(--primary);
      box-shadow: 0 0 0 2px rgba(14, 165, 233, 0.2);
    }
    .action-bar {
      display: flex;
      justify-content: center;
      margin-bottom: 28px;
    }
    .btn-submit {
      background: var(--primary-gradient);
      color: #fff;
      border: none;
      border-radius: 12px;
      padding: 14px 40px;
      font-size: 15px;
      font-weight: 600;
      cursor: pointer;
      transition: transform 0.2s, box-shadow 0.2s;
      display: flex;
      align-items: center;
      gap: 8px;
      box-shadow: 0 4px 20px rgba(99, 102, 241, 0.3);
    }
    .btn-submit:hover {
      transform: translateY(-2px);
      box-shadow: 0 6px 25px rgba(99, 102, 241, 0.45);
    }
    .btn-submit:disabled {
      opacity: 0.6;
      cursor: not-allowed;
      transform: none;
    }
    .results-container {
      display: none;
      background: var(--bg-surface);
      border: 1px solid var(--border-card);
      border-radius: 18px;
      padding: 24px;
      margin-bottom: 28px;
      animation: fadeIn 0.3s ease;
    }
    @keyframes fadeIn {
      from { opacity: 0; transform: translateY(10px); }
      to { opacity: 1; transform: translateY(0); }
    }
    .results-header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 20px;
      flex-wrap: wrap;
      gap: 12px;
    }
    .results-title {
      font-family: 'Outfit', sans-serif;
      font-size: 18px;
      font-weight: 700;
    }
    .risk-banner {
      padding: 10px 24px;
      border-radius: 9999px;
      font-size: 16px;
      font-weight: 700;
      letter-spacing: 0.5px;
      text-transform: uppercase;
      display: inline-flex;
      align-items: center;
      gap: 8px;
    }
    .risk-banner.low {
      background: var(--risk-low-bg);
      color: var(--risk-low);
      border: 1px solid var(--risk-low);
    }
    .risk-banner.moderate {
      background: var(--risk-mod-bg);
      color: var(--risk-mod);
      border: 1px solid var(--risk-mod);
    }
    .risk-banner.high {
      background: var(--risk-high-bg);
      color: var(--risk-high);
      border: 1px solid var(--risk-high);
    }
    .prob-grid {
      display: grid;
      grid-template-columns: repeat(3, 1fr);
      gap: 16px;
      margin-bottom: 20px;
    }
    .prob-box {
      background: rgba(30, 41, 59, 0.5);
      border: 1px solid var(--border-card);
      border-radius: 12px;
      padding: 14px;
      display: flex;
      flex-direction: column;
      gap: 6px;
    }
    .prob-header {
      display: flex;
      justify-content: space-between;
      font-size: 13px;
      font-weight: 600;
    }
    .prob-bar-track {
      height: 8px;
      background: rgba(255, 255, 255, 0.08);
      border-radius: 9999px;
      overflow: hidden;
    }
    .prob-bar-fill {
      height: 100%;
      border-radius: 9999px;
      transition: width 0.6s cubic-bezier(0.4, 0, 0.2, 1);
    }
    .fill-low { background: var(--risk-low); }
    .fill-mod { background: var(--risk-mod); }
    .fill-high { background: var(--risk-high); }
    .disclaimer-alert {
      background: rgba(245, 158, 11, 0.08);
      border: 1px solid rgba(245, 158, 11, 0.25);
      border-radius: 10px;
      padding: 12px 16px;
      font-size: 12px;
      color: #fbbf24;
      line-height: 1.4;
    }
    .error-alert {
      display: none;
      background: rgba(239, 68, 68, 0.1);
      border: 1px solid rgba(239, 68, 68, 0.3);
      border-radius: 10px;
      padding: 12px 16px;
      color: #fca5a5;
      font-size: 13px;
      margin-bottom: 20px;
    }
  </style>
</head>
<body>
  <div class="container">
    <header>
      <div>
        <div class="brand-title">
          <span>🩺</span> OncoRisk AI
        </div>
        <div class="brand-subtitle">
          Stage 1 Precision Oncology Toxicity Risk Inference Interface (Candidate Model V4)
        </div>
      </div>
      <div class="header-pills">
        <div class="badge">
          <span class="badge-dot"></span>
          <span>Frozen V4 Engine</span>
        </div>
        <a href="/docs" target="_blank" class="docs-link">Interactive API Docs →</a>
      </div>
    </header>

    <div class="preset-bar">
      <span class="preset-label">Quick Patient Presets:</span>
      <button type="button" class="btn-preset low" onclick="loadPreset('low')" id="btn-preset-low">Preset: Low Risk Patient</button>
      <button type="button" class="btn-preset mod" onclick="loadPreset('moderate')" id="btn-preset-mod">Preset: Moderate Risk Patient</button>
      <button type="button" class="btn-preset high" onclick="loadPreset('high')" id="btn-preset-high">Preset: High Risk Patient</button>
      <button type="button" class="btn-preset" onclick="resetForm()" id="btn-reset">Reset Form</button>
    </div>

    <div class="error-alert" id="error-alert"></div>

    <div class="results-container" id="results-container">
      <div class="results-header">
        <div>
          <div class="results-title">Toxicity Risk Assessment</div>
          <span style="font-size: 12px; color: var(--text-muted);">Evaluated by Frozen Candidate Model V4</span>
        </div>
        <div class="risk-banner" id="risk-banner">LOW</div>
      </div>

      <div class="prob-grid">
        <div class="prob-box">
          <div class="prob-header">
            <span>Low Risk</span>
            <span id="pct-low">0%</span>
          </div>
          <div class="prob-bar-track">
            <div class="prob-bar-fill fill-low" id="bar-low" style="width: 0%;"></div>
          </div>
        </div>
        <div class="prob-box">
          <div class="prob-header">
            <span>Moderate Risk</span>
            <span id="pct-mod">0%</span>
          </div>
          <div class="prob-bar-track">
            <div class="prob-bar-fill fill-mod" id="bar-mod" style="width: 0%;"></div>
          </div>
        </div>
        <div class="prob-box">
          <div class="prob-header">
            <span>High Risk</span>
            <span id="pct-high">0%</span>
          </div>
          <div class="prob-bar-track">
            <div class="prob-bar-fill fill-high" id="bar-high" style="width: 0%;"></div>
          </div>
        </div>
      </div>
    </div>

    <form id="toxicity-form" onsubmit="handlePredict(event)">
      <div class="grid-cards">
        <!-- Card 1: Demographics & Vitals -->
        <div class="card">
          <div class="card-title"><span>👤</span> Demographics & Vitals</div>
          <div class="form-group">
            <label for="age">Age (Years)</label>
            <input type="number" step="1" id="age" name="age" required min="18" max="120" value="62">
          </div>
          <div class="form-group">
            <label for="sex">Sex</label>
            <select id="sex" name="sex" required>
              <option value="Female">Female</option>
              <option value="Male">Male</option>
            </select>
          </div>
          <div class="form-group">
            <label for="heart_rate">Heart Rate (BPM)</label>
            <input type="number" step="1" id="heart_rate" name="heart_rate" required value="78">
          </div>
          <div class="form-group">
            <label for="systolic_bp">Systolic BP (mmHg)</label>
            <input type="number" step="1" id="systolic_bp" name="systolic_bp" required value="128">
          </div>
          <div class="form-group">
            <label for="diastolic_bp">Diastolic BP (mmHg)</label>
            <input type="number" step="1" id="diastolic_bp" name="diastolic_bp" required value="80">
          </div>
          <div class="form-group">
            <label for="oxygen_saturation">Oxygen Saturation SpO2 (%)</label>
            <input type="number" step="any" id="oxygen_saturation" name="oxygen_saturation" required value="96.5">
          </div>
          <div class="form-group">
            <label for="smoking_history">Smoking History</label>
            <select id="smoking_history" name="smoking_history" required>
              <option value="Never">Never</option>
              <option value="Former" selected>Former</option>
              <option value="Current">Current</option>
            </select>
          </div>
        </div>

        <!-- Card 2: Oncology & Genomics -->
        <div class="card">
          <div class="card-title"><span>🧬</span> Oncology & Genomics</div>
          <div class="form-group">
            <label for="cancer_type">Cancer Classification</label>
            <select id="cancer_type" name="cancer_type" required>
              <option value="NSCLC" selected>NSCLC (Non-Small Cell Lung)</option>
              <option value="Breast Cancer">Breast Cancer</option>
              <option value="Colorectal">Colorectal Cancer</option>
              <option value="Prostate">Prostate Cancer</option>
              <option value="Other">Other</option>
            </select>
          </div>
          <div class="form-group">
            <label for="cancer_stage">Cancer Stage</label>
            <select id="cancer_stage" name="cancer_stage" required>
              <option value="Stage I">Stage I</option>
              <option value="Stage II">Stage II</option>
              <option value="Stage III" selected>Stage III</option>
              <option value="Stage IV">Stage IV</option>
            </select>
          </div>
          <div class="form-group">
            <label for="mutation_primary">Primary Mutation</label>
            <select id="mutation_primary" name="mutation_primary" required>
              <option value="EGFR" selected>EGFR</option>
              <option value="KRAS">KRAS</option>
              <option value="BRAF">BRAF</option>
              <option value="PIK3CA">PIK3CA</option>
              <option value="None">None</option>
            </select>
          </div>
          <div class="form-group">
            <label for="mutation_secondary">Secondary Mutation</label>
            <select id="mutation_secondary" name="mutation_secondary" required>
              <option value="TP53" selected>TP53</option>
              <option value="PIK3CA">PIK3CA</option>
              <option value="PTEN">PTEN</option>
              <option value="None">None</option>
            </select>
          </div>
          <div class="form-group">
            <label for="mutation_burden">Tumor Mutation Burden (TMB)</label>
            <input type="number" step="any" id="mutation_burden" name="mutation_burden" required value="6.8">
          </div>
          <div class="form-group">
            <label for="gene_expression_score">Gene Expression Score</label>
            <input type="number" step="any" id="gene_expression_score" name="gene_expression_score" required value="49.8">
          </div>
          <div class="form-group">
            <label for="ctdna_level">ctDNA Level (ng/mL)</label>
            <input type="number" step="any" id="ctdna_level" name="ctdna_level" required value="1.4">
          </div>
        </div>

        <!-- Card 3: Biomarkers & Laboratory -->
        <div class="card">
          <div class="card-title"><span>🧪</span> Biomarkers & Laboratory</div>
          <div class="form-group">
            <label for="hemoglobin">Hemoglobin (g/dL)</label>
            <input type="number" step="any" id="hemoglobin" name="hemoglobin" required value="12.5">
          </div>
          <div class="form-group">
            <label for="white_blood_cell_count">WBC Count (k/uL)</label>
            <input type="number" step="any" id="white_blood_cell_count" name="white_blood_cell_count" required value="7.41">
          </div>
          <div class="form-group">
            <label for="platelet_count">Platelet Count (k/uL)</label>
            <input type="number" step="any" id="platelet_count" name="platelet_count" required value="231">
          </div>
          <div class="form-group">
            <label for="creatinine_level">Creatinine (mg/dL)</label>
            <input type="number" step="any" id="creatinine_level" name="creatinine_level" required value="1.0">
          </div>
          <div class="form-group">
            <label for="liver_function_marker">Liver Marker ALT/AST (U/L)</label>
            <input type="number" step="any" id="liver_function_marker" name="liver_function_marker" required value="17.3">
          </div>
          <div class="form-group">
            <label for="inflammation_marker">Inflammation CRP (mg/L)</label>
            <input type="number" step="any" id="inflammation_marker" name="inflammation_marker" required value="4.2">
          </div>
          <div class="form-group">
            <label for="tumor_marker_level">Tumor Marker (ng/mL)</label>
            <input type="number" step="any" id="tumor_marker_level" name="tumor_marker_level" required value="24.7">
          </div>
          <div class="form-group">
            <label for="biomarker_trend">Biomarker Trend</label>
            <select id="biomarker_trend" name="biomarker_trend" required>
              <option value="Stable" selected>Stable</option>
              <option value="Increasing">Increasing</option>
              <option value="Decreasing">Decreasing</option>
            </select>
          </div>
        </div>

        <!-- Card 4: Treatment Regimen & History -->
        <div class="card">
          <div class="card-title"><span>💊</span> Treatment & Clinical History</div>
          <div class="form-group">
            <label for="treatment_type">Treatment Type</label>
            <select id="treatment_type" name="treatment_type" required>
              <option value="Targeted Therapy" selected>Targeted Therapy</option>
              <option value="Chemotherapy">Chemotherapy</option>
              <option value="Immunotherapy">Immunotherapy</option>
              <option value="Hormone Therapy">Hormone Therapy</option>
            </select>
          </div>
          <div class="form-group">
            <label for="drug_name">Drug Name</label>
            <select id="drug_name" name="drug_name" required>
              <option value="Erlotinib" selected>Erlotinib</option>
              <option value="Cisplatin">Cisplatin</option>
              <option value="Pembrolizumab">Pembrolizumab</option>
              <option value="Trastuzumab">Trastuzumab</option>
              <option value="Oxaliplatin">Oxaliplatin</option>
            </select>
          </div>
          <div class="form-group">
            <label for="drug_dose">Drug Dose (mg)</label>
            <input type="number" step="any" id="drug_dose" name="drug_dose" required value="150">
          </div>
          <div class="form-group">
            <label for="treatment_cycle">Treatment Cycle Number</label>
            <input type="number" step="1" id="treatment_cycle" name="treatment_cycle" required value="4">
          </div>
          <div class="form-group">
            <label for="previous_treatment_count">Prior Treatment Regimens</label>
            <input type="number" step="1" id="previous_treatment_count" name="previous_treatment_count" required value="1">
          </div>
          <div class="form-group">
            <label for="previous_adverse_event">Prior Adverse Event</label>
            <select id="previous_adverse_event" name="previous_adverse_event" required>
              <option value="false" selected>No (False)</option>
              <option value="true">Yes (True)</option>
            </select>
          </div>
          <div class="form-group">
            <label for="previous_toxicity_grade">Prior Toxicity Grade (0.0 - 5.0)</label>
            <input type="number" step="any" id="previous_toxicity_grade" name="previous_toxicity_grade" required min="0" max="5" value="0.0">
          </div>
          <div class="form-group">
            <label for="comorbidity_count">Comorbidity Count</label>
            <input type="number" step="any" id="comorbidity_count" name="comorbidity_count" required value="1">
          </div>
        </div>
      </div>

      <div class="action-bar">
        <button type="submit" class="btn-submit" id="btn-analyze">
          <span>⚡</span> Analyze Toxicity Risk (Candidate V4)
        </button>
      </div>
    </form>
  </div>

  <script>
    const presets = {
      low: {
        age: 52, sex: "Female", heart_rate: 72, systolic_bp: 118, diastolic_bp: 76, oxygen_saturation: 98, smoking_history: "Never",
        cancer_type: "Breast Cancer", cancer_stage: "Stage II", mutation_primary: "PIK3CA", mutation_secondary: "None",
        mutation_burden: 3.2, gene_expression_score: 35.0, ctdna_level: 0.4,
        hemoglobin: 13.8, white_blood_cell_count: 6.2, platelet_count: 260, creatinine_level: 0.8, liver_function_marker: 18.0,
        inflammation_marker: 1.5, tumor_marker_level: 12.0, biomarker_trend: "Stable",
        treatment_type: "Targeted Therapy", drug_name: "Trastuzumab", drug_dose: 120, treatment_cycle: 2,
        previous_treatment_count: 0, previous_adverse_event: "false", previous_toxicity_grade: 0.0, comorbidity_count: 0
      },
      moderate: {
        age: 61, sex: "Male", heart_rate: 88, systolic_bp: 127, diastolic_bp: 65, oxygen_saturation: 90, smoking_history: "Former",
        cancer_type: "NSCLC", cancer_stage: "Stage III", mutation_primary: "KRAS", mutation_secondary: "None",
        mutation_burden: 0.11, gene_expression_score: 47.32, ctdna_level: 0.11,
        hemoglobin: 11.0, white_blood_cell_count: 8.67, platelet_count: 114, creatinine_level: 1.01, liver_function_marker: 5.0,
        inflammation_marker: 6.77, tumor_marker_level: 0.77, biomarker_trend: "Stable",
        treatment_type: "Immunotherapy", drug_name: "Pembrolizumab", drug_dose: 148, treatment_cycle: 1,
        previous_treatment_count: 0, previous_adverse_event: "true", previous_toxicity_grade: 0.0, comorbidity_count: 0
      },
      high: {
        age: 76, sex: "Male", heart_rate: 98, systolic_bp: 154, diastolic_bp: 96, oxygen_saturation: 92, smoking_history: "Current",
        cancer_type: "NSCLC", cancer_stage: "Stage IV", mutation_primary: "EGFR", mutation_secondary: "TP53",
        mutation_burden: 14.8, gene_expression_score: 72.0, ctdna_level: 5.8,
        hemoglobin: 9.4, white_blood_cell_count: 12.4, platelet_count: 130, creatinine_level: 2.4, liver_function_marker: 88.0,
        inflammation_marker: 14.2, tumor_marker_level: 95.0, biomarker_trend: "Increasing",
        treatment_type: "Chemotherapy", drug_name: "Cisplatin", drug_dose: 250, treatment_cycle: 6,
        previous_treatment_count: 3, previous_adverse_event: "true", previous_toxicity_grade: 3.0, comorbidity_count: 4
      }
    };

    function loadPreset(name) {
      const data = presets[name];
      if (!data) return;
      for (const [key, val] of Object.entries(data)) {
        const el = document.getElementById(key);
        if (el) el.value = val;
      }
    }

    function resetForm() {
      document.getElementById('toxicity-form').reset();
      document.getElementById('results-container').style.display = 'none';
      document.getElementById('error-alert').style.display = 'none';
    }

    async function handlePredict(e) {
      e.preventDefault();
      const errEl = document.getElementById('error-alert');
      const btn = document.getElementById('btn-analyze');
      const resContainer = document.getElementById('results-container');
      
      errEl.style.display = 'none';
      btn.disabled = true;
      btn.innerHTML = '<span>⏳</span> Analyzing Encounter...';

      const formData = new FormData(document.getElementById('toxicity-form'));
      const payload = {};
      formData.forEach((value, key) => {
        if (['age', 'mutation_burden', 'gene_expression_score', 'ctdna_level', 'tumor_marker_level',
             'inflammation_marker', 'heart_rate', 'systolic_bp', 'diastolic_bp', 'oxygen_saturation',
             'hemoglobin', 'white_blood_cell_count', 'platelet_count', 'creatinine_level',
             'liver_function_marker', 'drug_dose', 'previous_toxicity_grade', 'comorbidity_count'].includes(key)) {
          payload[key] = parseFloat(value);
        } else if (['treatment_cycle', 'previous_treatment_count'].includes(key)) {
          payload[key] = parseInt(value, 10);
        } else if (key === 'previous_adverse_event') {
          payload[key] = (value === 'true');
        } else {
          payload[key] = value;
        }
      });

      try {
        const res = await fetch('/predict', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(payload)
        });

        if (!res.ok) {
          const errData = await res.json();
          throw new Error(errData.details ? errData.details.join('; ') : (errData.detail || 'Prediction request failed.'));
        }

        const data = await res.json();
        renderResults(data);
      } catch (err) {
        errEl.innerText = '⚠️ Error: ' + err.message;
        errEl.style.display = 'block';
        resContainer.style.display = 'none';
      } finally {
        btn.disabled = false;
        btn.innerHTML = '<span>⚡</span> Analyze Toxicity Risk (Candidate V4)';
      }
    }

    function renderResults(data) {
      const resContainer = document.getElementById('results-container');
      const banner = document.getElementById('risk-banner');
      const risk = data.predicted_toxicity_risk;

      banner.className = 'risk-banner ' + risk.toLowerCase();
      banner.innerText = risk + ' Toxicity Risk';

      const probs = data.probabilities || {};
      const lowPct = Math.round((probs.Low || 0) * 100);
      const modPct = Math.round((probs.Moderate || 0) * 100);
      const highPct = Math.round((probs.High || 0) * 100);

      document.getElementById('pct-low').innerText = lowPct + '%';
      document.getElementById('bar-low').style.width = lowPct + '%';

      document.getElementById('pct-mod').innerText = modPct + '%';
      document.getElementById('bar-mod').style.width = modPct + '%';

      document.getElementById('pct-high').innerText = highPct + '%';
      document.getElementById('bar-high').style.width = highPct + '%';

      resContainer.style.display = 'block';
      resContainer.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }
  </script>
</body>
</html>
"""
