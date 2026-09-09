// Stage 3 Clinical NLP & Data Augmentation Platform - Client Application

document.addEventListener('DOMContentLoaded', () => {
  initTabs();
  initDiffViewer();
  fetchLiveMetrics();
});

// Tab Switching
function initTabs() {
  const tabs = document.querySelectorAll('.nav-tab');
  const contents = document.querySelectorAll('.tab-content');

  tabs.forEach(tab => {
    tab.addEventListener('click', () => {
      const targetId = tab.getAttribute('data-tab');
      tabs.forEach(t => t.classList.remove('active'));
      contents.forEach(c => c.classList.remove('active'));

      tab.classList.add('active');
      const targetContent = document.getElementById(targetId);
      if (targetContent) {
        targetContent.classList.add('active');
      }
    });
  });
}

// Clinical Samples Database for Diff Viewer
const CLINICAL_SAMPLES = {
  "doc-0": {
    title: "Oncology Consultation Progress Note (DOC-000004)",
    leftMeta: "Urgency: HIGH | Hazard: NONE | Split: TRAIN",
    rightMeta: "Method: Terminology + Lab Permutation (AUG-DOC-000004)",
    leftHtml: `ONCOLOGY CONSULTATION PROGRESS NOTE
Patient ID: PT-000002
Demographics: 66-year-old male presenting for Cycle 1 evaluation.

DIAGNOSIS & MOLECULAR PROFILING:
Primary Diagnosis: Stage III Unknown.
Genomic Profile: Confirmed driver mutation in <span class="ent-gene">TP53</span>. Secondary mutation: None/Unknown.
Tumor mutational burden measured at 5.2 mut/Mb with ctDNA level of 2.18 ng/mL.

CLINICAL ASSESSMENT & LABORATORY REVIEW:
Vitals: Blood pressure 134/68 mmHg, Heart rate 67 bpm, SpO2 98% on room air.
Serum Chemistries: Serum creatinine 1.49 mg/dL, Liver function enzymes 17.3 U/L, Hemoglobin 12.4 g/dL.
Patient currently demonstrates <span class="ent-adverse">moderate fatigue and mild nausea</span>. Physical examination reveals no jaundice, no acute respiratory distress, and clear lung fields bilaterally.

TREATMENT PLAN & REGIMEN:
Administer scheduled therapy with <span class="ent-drug">radiotherapy-standard</span> at a dosage of <span class="ent-dosage">181.9 mg</span>.
Pre-medications ordered per protocol. Serial monitoring of renal markers and complete blood count scheduled prior to next infusion.`,
    rightHtml: `ONCOLOGY CONSULTATION PROGRESS NOTE
Patient ID: PT-000002
Demographics: <span class="carrier-diff">66-year-old male evaluated prior to planned Cycle</span> 1 evaluation.

DIAGNOSIS & MOLECULAR PROFILING:
Primary Diagnosis: Stage III Unknown.
Genomic Profile: Confirmed driver mutation in <span class="ent-gene">TP53</span>. Secondary mutation: None/Unknown.
Tumor mutational burden measured at 5.2 mut/Mb with ctDNA level of 2.18 ng/mL.

<span class="carrier-diff">OBJECTIVE FINDINGS & LAB REVIEW:</span>
Vitals: <span class="carrier-diff">Heart rate 67 bpm, Blood pressure 134/68 mmHg</span>, SpO2 98% on room air.
Serum Chemistries: <span class="carrier-diff">Liver function enzymes 17.3 U/L, Serum creatinine 1.49 mg/dL</span>, Hemoglobin 12.4 g/dL.
Patient currently demonstrates <span class="ent-adverse">moderate fatigue and mild nausea</span>. <span class="carrier-diff">Clinical examination shows</span> no jaundice, no acute respiratory distress, and <span class="carrier-diff">lungs clear to auscultation bilaterally</span>.

<span class="carrier-diff">PLANNED THERAPY & ONCOLOGY REGIMEN:</span>
Administer scheduled therapy with <span class="ent-drug">radiotherapy-standard</span> at a dosage of <span class="ent-dosage">181.9 mg</span>.
<span class="carrier-diff">Pre-treatment medications administered per standard protocol</span>. <span class="carrier-diff">Routine surveillance of renal biomarkers</span> and <span class="carrier-diff">CBC scheduled before subsequent infusion</span>.`
  },
  "doc-1": {
    title: "Surgical Pathology Report (DOC-000005)",
    leftMeta: "Urgency: LOW | Hazard: NONE | Split: TRAIN",
    rightMeta: "Method: Pathological Synonyms (AUG-DOC-000005)",
    leftHtml: `SURGICAL PATHOLOGY REPORT
Patient ID: PT-000002
Specimen: Core needle biopsy / surgical tissue resection of primary tumor.

GROSS & MICROSCOPIC EXAMINATION:
Specimen consists of multiple fibrous tissue cores exhibiting invasive malignant epithelial neoplasm consistent with <span class="ent-adverse">Unknown</span>.
Microscopic sections reveal moderate cellular atypia with hyperchromatic nuclei, prominent nucleoli, and desmoplastic stroma.
Surgical margins are evaluated and confirmed clear of invasive tumor cells. Lymphovascular invasion is not identified.

MOLECULAR & IMMUNOHISTOCHEMICAL TESTING:
Next-Generation Sequencing (NGS) confirms activating <span class="ent-gene">TP53</span> driver alteration.
Pathologic Staging: Pathologic evaluation aligns with clinical Stage III.
Recommendation: Correlation with systemic clinical oncology records and targeted therapy protocols.`,
    rightHtml: `SURGICAL PATHOLOGY REPORT
Patient ID: PT-000002
Specimen: Core needle biopsy / surgical tissue resection of primary tumor.

<span class="carrier-diff">PATHOLOGIC & HISTOLOGIC EXAMINATION:</span>
<span class="carrier-diff">Specimen demonstrates multiple fibrous tissue cores showing</span> invasive malignant epithelial neoplasm consistent with <span class="ent-adverse">Unknown</span>.
<span class="carrier-diff">Histologic sections demonstrate</span> moderate cellular atypia with hyperchromatic nuclei, prominent nucleoli, and desmoplastic stroma.
Surgical margins are evaluated and confirmed clear of invasive tumor cells. Lymphovascular invasion is not identified.

MOLECULAR & IMMUNOHISTOCHEMICAL TESTING:
Next-Generation Sequencing (NGS) confirms activating <span class="ent-gene">TP53</span> driver alteration.
Pathologic Staging: <span class="carrier-diff">Histopathologic findings align with</span> clinical Stage III.
Recommendation: <span class="carrier-diff">Clinical correlation with outpatient oncology documentation</span> and targeted therapy protocols.`
  },
  "doc-2": {
    title: "Ambulatory Nurse Intake Assessment (DOC-000006)",
    leftMeta: "Urgency: LOW | Hazard: NONE | Split: TRAIN",
    rightMeta: "Method: Vitals Permutation + Nursing Synonyms (AUG-DOC-000006)",
    leftHtml: `AMBULATORY ONCOLOGY NURSE INTAKE ASSESSMENT
Patient ID: PT-000002 | Infusion Cycle: 1

TRIAGE & CLINICAL VITALS:
Vital Signs: BP 134/68 mmHg, Pulse 67 bpm, Resp Rate 18/min, SpO2 98%.
ECOG Performance Status: Evaluated as ambulatory and self-caring.

SYMPTOM REVIEW & ADVERSE EVENT SCREENING:
Patient screened prior to administration of <span class="ent-drug">radiotherapy-standard</span> at <span class="ent-dosage">181.9 mg</span>.
Patient reports <span class="ent-adverse">mild treatment-related fatigue</span>. Patient explicitly denies chest pain, denies intractable vomiting, and reports no shaking chills.
IV peripheral access established with blood return confirmed in right forearm.

NURSING INTERVENTIONS:
Hydration protocol initiated per oncology standing orders. Physician notified regarding baseline labs and symptom status.`,
    rightHtml: `AMBULATORY ONCOLOGY NURSE INTAKE ASSESSMENT
Patient ID: PT-000002 | Infusion Cycle: 1

<span class="carrier-diff">VITAL SIGNS & INTAKE TRIAGE:</span>
Vital Signs: <span class="carrier-diff">Pulse 67 bpm, BP 134/68 mmHg, Resp Rate 18/min, SpO2 98%</span>.
ECOG Performance Status: <span class="carrier-diff">Assessed as ambulatory and capable of self-care</span>.

SYMPTOM REVIEW & ADVERSE EVENT SCREENING:
Patient screened prior to administration of <span class="ent-drug">radiotherapy-standard</span> at <span class="ent-dosage">181.9 mg</span>.
Patient reports <span class="ent-adverse">mild treatment-related fatigue</span>. Patient explicitly denies chest pain, denies intractable vomiting, and reports no shaking chills.
<span class="carrier-diff">Peripheral IV line established with brisk blood return in right forearm</span>.

<span class="carrier-diff">CLINICAL NURSING ACTIONS:</span>
<span class="carrier-diff">Pre-treatment IV hydration administered per oncology unit protocol</span>. <span class="carrier-diff">Supervising oncologist notified of baseline labs and triage status</span>.`
  },
  "doc-3": {
    title: "Patient Symptom Log & Daily Report",
    leftMeta: "Urgency: MEDIUM | Hazard: HEPATIC | Split: TRAIN",
    rightMeta: "Method: Conversational Framing Variation",
    leftHtml: `PATIENT SYMPTOM LOG & CALL-IN ASSESSMENT
Patient ID: PT-000045 | Log Day: 14

PATIENT REPORTED OUTCOMES:
I am reporting my symptoms for cycle 2. Overall I am feeling unwell with <span class="ent-adverse">right upper quadrant abdominal discomfort</span> and noticed <span class="ent-adverse">dark urine</span> since yesterday morning.
I took my prescribed medicine as directed with oral <span class="ent-drug">Osimertinib</span> at <span class="ent-dosage">80 mg</span> daily.
Patient explicitly denies high fever, denies vomiting, and denies chest pressure.
Will contact the clinic if symptoms worsen.`,
    rightHtml: `PATIENT SYMPTOM LOG & CALL-IN ASSESSMENT
Patient ID: PT-000045 | Log Day: 14

PATIENT REPORTED OUTCOMES:
<span class="carrier-diff">Logging my daily symptoms for</span> cycle 2. <span class="carrier-diff">In general I am feeling</span> unwell with <span class="ent-adverse">right upper quadrant abdominal discomfort</span> and noticed <span class="ent-adverse">dark urine</span> since yesterday morning.
<span class="carrier-diff">I took my prescribed medications as instructed</span> with oral <span class="ent-drug">Osimertinib</span> at <span class="ent-dosage">80 mg</span> daily.
Patient explicitly denies high fever, denies vomiting, and denies chest pressure.
<span class="carrier-diff">Will alert the oncology triage desk if symptoms worsen</span>.`
  }
};

function initDiffViewer() {
  const selector = document.getElementById('diff-doc-selector');
  if (!selector) return;

  function updateView(key) {
    const data = CLINICAL_SAMPLES[key];
    if (!data) return;

    document.getElementById('diff-left-title').textContent = data.title + ' — Source';
    document.getElementById('diff-left-meta').textContent = data.leftMeta;
    document.getElementById('diff-left-text').innerHTML = data.leftHtml;

    document.getElementById('diff-right-title').textContent = data.title + ' — Augmented';
    document.getElementById('diff-right-meta').textContent = data.rightMeta;
    document.getElementById('diff-right-text').innerHTML = data.rightHtml;
  }

  selector.addEventListener('change', (e) => {
    updateView(e.target.value);
  });

  // Load initial sample
  updateView(selector.value);
}

// Fetch live metrics from local server if active
function fetchLiveMetrics() {
  fetch('/api/data')
    .then(r => r.json())
    .then(data => {
      if (data.diversity) {
        console.log('Loaded live diversity metrics:', data.diversity);
      }
    })
    .catch(() => {
      console.log('Running in static mode with embedded verified data.');
    });
}
