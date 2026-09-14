export const patient = { id: "ONC-2048", age: 62, sex: "Male", cancer: "NSCLC", stage: "IV", ecog: 1, treatment: "Carboplatin + Pemetrexed", creatinine: 1.8, hemoglobin: 9.8, platelets: 140, cea: 27, ctdna: "Increasing" };
export const metrics = [
  { key: "ML", label: "Toxicity Risk", value: "72%", status: "HIGH", tone: "red" },
  { key: "DL", label: "Progression Risk", value: "81%", status: "HIGH", tone: "violet" },
  { key: "NLP", label: "Clinical Urgency", value: "HIGH", status: "REVIEW", tone: "cyan" },
  { key: "SLM", label: "Guidance", value: "SAFE", status: "GROUNDED", tone: "green" },
];
export const trend = [{ m: "Jan", ctDNA: 24, CEA: 15 }, { m: "Feb", ctDNA: 29, CEA: 17 }, { m: "Mar", ctDNA: 37, CEA: 18 }, { m: "Apr", ctDNA: 49, CEA: 21 }, { m: "May", ctDNA: 61, CEA: 24 }, { m: "Jun", ctDNA: 76, CEA: 27 }];
export const factors = [{ name: "Creatinine", value: 24 }, { name: "Treatment", value: 18 }, { name: "Age", value: 12 }, { name: "Hemoglobin", value: 9 }, { name: "Platelets", value: 7 }];
export const roc = [{ x: 0, ML: 0, DL: 0, NLP: 0 }, { x: .08, ML: .48, DL: .55, NLP: .6 }, { x: .2, ML: .7, DL: .76, NLP: .81 }, { x: .4, ML: .86, DL: .88, NLP: .91 }, { x: .65, ML: .95, DL: .96, NLP: .97 }, { x: 1, ML: 1, DL: 1, NLP: 1 }];
