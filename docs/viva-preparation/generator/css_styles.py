CSS_CONTENT = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=Fira+Code:wght@400;500&display=swap');

:root {
  --primary: #1e3a8a;
  --primary-light: #3b82f6;
  --primary-dark: #0f172a;
  --secondary: #0284c7;
  --accent: #059669;
  --accent-light: #ecfdf5;
  --accent-border: #10b981;
  --warning-bg: #fffbeb;
  --warning-border: #f59e0b;
  --warning-text: #92400e;
  --info-bg: #eff6ff;
  --info-border: #3b82f6;
  --info-text: #1e40af;
  --card-bg: #ffffff;
  --bg: #f8fafc;
  --text-main: #1e293b;
  --text-muted: #64748b;
  --border: #e2e8f0;
}

* {
  box-sizing: border-box;
  margin: 0;
  padding: 0;
}

body {
  font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
  color: var(--text-main);
  background-color: var(--bg);
  line-height: 1.6;
  font-size: 13.5px;
}

@page {
  size: A4 portrait;
  margin: 14mm 12mm 14mm 12mm;
  @bottom-center {
    content: "Page " counter(page);
    font-family: 'Inter', sans-serif;
    font-size: 8pt;
    color: #94a3b8;
  }
}

.page-break {
  page-break-after: always;
  break-after: page;
}

.avoid-break {
  break-inside: avoid;
  page-break-inside: avoid;
}

.container {
  max-width: 960px;
  margin: 0 auto;
  padding: 16px;
  background: #ffffff;
}

/* Header & Cover Banner */
.cover-header {
  background: linear-gradient(135deg, #0f172a 0%, #1e3a8a 50%, #0369a1 100%);
  color: #ffffff;
  padding: 32px 28px;
  border-radius: 12px;
  margin-bottom: 24px;
  box-shadow: 0 10px 25px -5px rgba(15, 23, 42, 0.25);
  border: 1px solid rgba(255, 255, 255, 0.1);
}

.cover-header h1 {
  font-size: 26px;
  font-weight: 800;
  letter-spacing: -0.5px;
  line-height: 1.25;
  margin-bottom: 6px;
  color: #ffffff;
}

.cover-header .subtitle {
  font-size: 15px;
  font-weight: 500;
  color: #93c5fd;
  margin-bottom: 14px;
}

.project-badge-bar {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
  margin-top: 14px;
  padding-top: 12px;
  border-top: 1px solid rgba(255, 255, 255, 0.15);
}

.badge {
  display: inline-block;
  padding: 3px 9px;
  border-radius: 9999px;
  font-size: 11px;
  font-weight: 600;
  letter-spacing: 0.3px;
  text-transform: uppercase;
}

.badge-dark {
  background: rgba(255, 255, 255, 0.15);
  color: #f1f5f9;
  border: 1px solid rgba(255, 255, 255, 0.2);
}

.badge-green {
  background: #059669;
  color: #ffffff;
}

.badge-blue {
  background: #2563eb;
  color: #ffffff;
}

.badge-amber {
  background: #d97706;
  color: #ffffff;
}

.project-context-box {
  background: #f0fdf4;
  border-left: 4px solid #16a34a;
  padding: 12px 16px;
  border-radius: 6px;
  margin-bottom: 24px;
  font-size: 12.5px;
  color: #14532d;
  line-height: 1.5;
}

.project-context-box strong {
  color: #166534;
}

/* Section Headings */
.part-title-banner {
  background: linear-gradient(90deg, #1e293b 0%, #334155 100%);
  color: #ffffff;
  padding: 12px 18px;
  border-radius: 8px;
  margin: 28px 0 16px 0;
  display: flex;
  align-items: center;
  justify-content: space-between;
  page-break-after: avoid;
  break-after: avoid;
}

.part-title-banner h2 {
  font-size: 17px;
  font-weight: 700;
  letter-spacing: -0.3px;
  text-transform: uppercase;
}

.section-intro {
  font-size: 13px;
  color: #475569;
  margin-bottom: 16px;
  font-style: italic;
}

/* Concept Cards */
.concept-card {
  background: #ffffff;
  border: 1px solid var(--border);
  border-radius: 10px;
  padding: 16px 18px;
  margin-bottom: 16px;
  box-shadow: 0 2px 5px rgba(0, 0, 0, 0.03);
  page-break-inside: avoid;
  break-inside: avoid;
}

.concept-card.highlight {
  border-left: 4px solid var(--primary-light);
}

.concept-header {
  display: flex;
  align-items: baseline;
  justify-content: space-between;
  margin-bottom: 10px;
  border-bottom: 1px solid #f1f5f9;
  padding-bottom: 6px;
}

.concept-title {
  font-size: 15px;
  font-weight: 700;
  color: var(--primary-dark);
}

.concept-num {
  font-size: 12px;
  font-weight: 700;
  color: var(--primary-light);
  background: #eff6ff;
  padding: 2px 7px;
  border-radius: 4px;
}

/* Content Blocks Inside Concept Card */
.def-block {
  margin-bottom: 10px;
  font-size: 13px;
  line-height: 1.55;
  color: #334155;
}

.def-block strong {
  color: #0f172a;
}

.how-block {
  background: #f8fafc;
  border: 1px dashed #cbd5e1;
  border-radius: 6px;
  padding: 8px 12px;
  margin-bottom: 10px;
  font-size: 12.5px;
  color: #334155;
}

.how-block strong {
  color: #1e293b;
}

.example-box {
  background: #fdf4ff;
  border-left: 3.5px solid #a855f7;
  padding: 9px 13px;
  border-radius: 0 6px 6px 0;
  margin-bottom: 10px;
  font-size: 12.5px;
  color: #581c87;
  line-height: 1.5;
}

.example-box strong {
  color: #6b21a8;
}

.viva-box {
  background: #ecfdf5;
  border-left: 3.5px solid #10b981;
  padding: 9px 13px;
  border-radius: 0 6px 6px 0;
  margin-bottom: 10px;
  font-size: 12.5px;
  color: #064e3b;
  line-height: 1.5;
}

.viva-box strong {
  color: #047857;
}

.remember-box {
  background: #fffbeb;
  border-left: 3.5px solid #f59e0b;
  padding: 8px 13px;
  border-radius: 0 6px 6px 0;
  font-size: 12px;
  color: #78350f;
  line-height: 1.45;
}

.remember-box strong {
  color: #b45309;
}

/* Tables */
.table-container {
  margin: 16px 0 24px 0;
  border-radius: 8px;
  overflow: hidden;
  border: 1px solid #cbd5e1;
  box-shadow: 0 2px 5px rgba(0,0,0,0.02);
  page-break-inside: avoid;
  break-inside: avoid;
}

table {
  width: 100%;
  border-collapse: collapse;
  font-size: 12px;
  background: #ffffff;
}

th {
  background: #1e293b;
  color: #ffffff;
  font-weight: 600;
  text-align: left;
  padding: 9px 12px;
  font-size: 12px;
  letter-spacing: 0.2px;
}

td {
  padding: 8px 12px;
  border-bottom: 1px solid #e2e8f0;
  color: #334155;
  vertical-align: top;
  line-height: 1.45;
}

tr:nth-child(even) td {
  background-color: #f8fafc;
}

tr:last-child td {
  border-bottom: none;
}

td strong {
  color: #0f172a;
}

/* Roles Specific Cards */
.role-card {
  background: #ffffff;
  border: 1px solid #cbd5e1;
  border-radius: 10px;
  padding: 18px 20px;
  margin-bottom: 22px;
  box-shadow: 0 3px 8px rgba(0,0,0,0.04);
  page-break-inside: avoid;
  break-inside: avoid;
}

.role-title {
  font-size: 17px;
  font-weight: 800;
  color: #1e3a8a;
  margin-bottom: 6px;
  display: flex;
  align-items: center;
  justify-content: space-between;
}

.role-tag {
  background: #e0e7ff;
  color: #3730a3;
  font-size: 11px;
  font-weight: 700;
  padding: 3px 8px;
  border-radius: 4px;
}

.role-section {
  margin-top: 10px;
  font-size: 12.5px;
  line-height: 1.5;
}

.role-section h4 {
  font-size: 13px;
  font-weight: 700;
  color: #0f172a;
  margin: 8px 0 3px 0;
}

.role-tools {
  background: #f1f5f9;
  padding: 6px 10px;
  border-radius: 5px;
  font-family: 'Fira Code', monospace;
  font-size: 11.5px;
  color: #0f172a;
  margin: 6px 0;
  display: inline-block;
}

/* Viva Questions Styles */
.qa-card {
  background: #ffffff;
  border: 1px solid #e2e8f0;
  border-radius: 8px;
  padding: 12px 16px;
  margin-bottom: 12px;
  page-break-inside: avoid;
  break-inside: avoid;
}

.qa-card:hover {
  border-color: #93c5fd;
}

.qa-q {
  font-size: 13px;
  font-weight: 700;
  color: #1e293b;
  margin-bottom: 6px;
  display: flex;
  align-items: flex-start;
  gap: 8px;
}

.q-num {
  background: #3b82f6;
  color: #ffffff;
  font-size: 11px;
  font-weight: 800;
  padding: 1px 6px;
  border-radius: 4px;
  flex-shrink: 0;
  margin-top: 1px;
}

.qa-a {
  font-size: 12.5px;
  color: #334155;
  line-height: 1.5;
  padding-left: 28px;
}

.qa-a strong {
  color: #0f172a;
}

/* Rapid Fire Grid */
.rapid-grid {
  display: grid;
  grid-template-columns: 1fr;
  gap: 10px;
  margin: 14px 0;
}

.rapid-item {
  background: #f8fafc;
  border: 1px solid #e2e8f0;
  border-left: 3px solid #0284c7;
  padding: 8px 12px;
  border-radius: 0 6px 6px 0;
  font-size: 12px;
  page-break-inside: avoid;
  break-inside: avoid;
}

.rapid-q {
  font-weight: 700;
  color: #0f172a;
  margin-bottom: 3px;
}

.rapid-a {
  color: #334155;
  line-height: 1.4;
}

/* Revision Section */
.rev-box {
  background: #ffffff;
  border: 1px solid #cbd5e1;
  border-radius: 8px;
  padding: 14px 18px;
  margin-bottom: 16px;
  page-break-inside: avoid;
  break-inside: avoid;
}

.rev-box h3 {
  font-size: 14px;
  font-weight: 800;
  color: #1e3a8a;
  border-bottom: 1.5px solid #e2e8f0;
  padding-bottom: 5px;
  margin-bottom: 10px;
}

.rev-list {
  list-style: none;
}

.rev-list li {
  position: relative;
  padding-left: 18px;
  margin-bottom: 6px;
  font-size: 12px;
  color: #334155;
  line-height: 1.45;
}

.rev-list li::before {
  content: "✔";
  position: absolute;
  left: 0;
  color: #10b981;
  font-weight: 800;
  font-size: 11px;
}

.rev-list li strong {
  color: #0f172a;
}
"""
