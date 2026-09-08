# Data Science Viva Preparation — Level 1
## Personalized Precision Medicine for Oncology Treatment Optimization

This directory contains the complete, student-friendly **Data Science Viva Level 1 Study Material**, specifically customized and grounded in our Precision Oncology AI system.

### 📚 Materials Included

1. **[Data_Science_Viva_Level_1_Study_Material.pdf](Data_Science_Viva_Level_1_Study_Material.pdf)**  
   *The primary, high-resolution, print-ready PDF study guide (A4 portrait, styled with badges, tables, and color-coded callouts).*

2. **[Data_Science_Viva_Level_1_Study_Material.html](Data_Science_Viva_Level_1_Study_Material.html)**  
   *Interactive standalone HTML version for quick viewing on laptops, tablets, or mobile devices.*

3. **[complete_project_specific_data_science_viva.pdf](complete_project_specific_data_science_viva.pdf)**  
   *Project-specific deep-dive reference PDF.*

4. **[generator/](generator/)**  
   *Modular, reproducible Python scripts that construct and compile the PDF and HTML materials using Microsoft Edge headless rendering:*
   - `make_pdf.py`: Master assembly and PDF compiler
   - `css_styles.py`: Responsive modern print/screen CSS stylesheet
   - `part1_ml.py`: 24 core Machine Learning concepts (Definition, Project Example, Viva Answer, Remember)
   - `part2_dl.py`: 27 core Deep Learning concepts (Definition, How it works, Project Example, Viva Answer, Remember)
   - `part3_roles.py`: 5 Data Science Engineering Roles (Data Eng, EDA Eng, ML Eng, Eval Eng, Integration Eng)
   - `part4_tables.py`: 13 comprehensive comparison tables
   - `part5_questions.py`: 95 curated viva questions & answers (Basic, Intermediate, Why, Practical, Rapid-Fire)
   - `part6_revision.py`: '1 Hour Before Viva' quick revision checklists & top 25 must-know questions

---

### 🚀 Re-generating the PDF
To re-generate the PDF at any time, run:
```bash
python docs/viva-preparation/generator/make_pdf.py
```
