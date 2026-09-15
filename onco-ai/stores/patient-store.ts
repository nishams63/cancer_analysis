import { create } from "zustand";
import { persist } from "zustand/middleware";

export interface Patient {
  id: string;
  age: number;
  sex: string;
  cancer: string;
  stage: string;
  ecog: number;
  treatment: string;
  creatinine: number;
  hemoglobin: number;
  platelets: number;
  cea: number;
  ctdna: string;
  status: "Review" | "Stable" | "Critical" | "Watch";
}

const initialPatients: Patient[] = [
  {
    id: "ONC-2048",
    age: 62,
    sex: "Male",
    cancer: "NSCLC",
    stage: "IV",
    ecog: 1,
    treatment: "Carboplatin + Pemetrexed",
    creatinine: 1.8,
    hemoglobin: 9.8,
    platelets: 140,
    cea: 27,
    ctdna: "Increasing",
    status: "Review",
  },
  {
    id: "ONC-2049",
    age: 57,
    sex: "Female",
    cancer: "Breast (HER2+)",
    stage: "III",
    ecog: 0,
    treatment: "Trastuzumab + Pertuzumab",
    creatinine: 0.9,
    hemoglobin: 12.4,
    platelets: 210,
    cea: 4.2,
    ctdna: "Undetectable",
    status: "Stable",
  },
  {
    id: "ONC-2050",
    age: 69,
    sex: "Male",
    cancer: "Colorectal",
    stage: "IV",
    ecog: 2,
    treatment: "FOLFOX + Bevacizumab",
    creatinine: 1.4,
    hemoglobin: 10.2,
    platelets: 165,
    cea: 48,
    ctdna: "Rising",
    status: "Critical",
  },
];

interface PatientState {
  patients: Patient[];
  activePatient: Patient;
  addPatient: (newPatient: Patient) => void;
  updatePatient: (patientId: string, updates: Partial<Patient>) => void;
  setActivePatient: (patientId: string) => void;
}

export const usePatientStore = create<PatientState>()(persist((set, get) => ({
  patients: initialPatients,
  activePatient: initialPatients[0],
  addPatient: (newPatient: Patient) => {
    set((state) => ({
      patients: [newPatient, ...state.patients],
      activePatient: newPatient,
    }));
  },
  updatePatient: (patientId: string, updates: Partial<Patient>) => {
    set((state) => {
      const patients = state.patients.map((patient) => patient.id === patientId ? { ...patient, ...updates } : patient);
      const activePatient = state.activePatient.id === patientId ? { ...state.activePatient, ...updates } : state.activePatient;
      return { patients, activePatient };
    });
  },
  setActivePatient: (patientId: string) => {
    const found = get().patients.find((p) => p.id.toUpperCase() === patientId.toUpperCase());
    if (found) {
      set({ activePatient: found });
    }
  },
}), {
  name: "onco-ai-patient-context",
  version: 1,
}));
