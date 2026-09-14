import { NextRequest, NextResponse } from "next/server";
import { getServerSession } from "next-auth";
import { authOptions } from "@/lib/auth/options";
import { callAIBackend } from "@/lib/ai-client";
import { clinicalInputSchema, decisionSchema, safetyInputSchema, slmInputSchema } from "@/lib/validation";

const windows = new Map<string, { count: number; reset: number }>();
function rateLimit(request: NextRequest) {
  const key = request.headers.get("x-forwarded-for")?.split(",")[0] || "local";
  const now = Date.now(), current = windows.get(key);
  if (!current || current.reset < now) { windows.set(key, { count: 1, reset: now + 60_000 }); return true; }
  if (current.count >= 40) return false;
  current.count += 1; return true;
}
function demo(route: string) {
  const createdAt = new Date().toISOString();
  if (route === "ml/predict") return { prediction: "high", probability: .72, confidence: .89, contributors: [{ feature: "creatinine", impact: .24 }, { feature: "treatment", impact: .18 }], model: "candidate-v4", model_version: "1.0", createdAt };
  if (route === "dl/predict") return { progression_probability: .81, confidence: .92, spatial_score: .77, temporal_score: .85, fusion_score: .81, model: "fusion-v3", createdAt };
  if (route === "nlp/analyze") return { urgency: "high", confidence: .91, entities: ["fatigue", "Carboplatin", "lymph node progression", "EGFR mutation"], negations: [{ text: "chest pain", state: "absent" }], summary: "High-urgency synthetic report requiring clinician review.", createdAt };
  if (route === "slm/chat") return { response: "Demo / simulated output: Elevated renal toxicity and progression signals warrant multidisciplinary clinician review.", evidence: [{ id: "DEMO-EV-2048", title: "Synthetic patient record" }], safety: { grounded: true, dosage_safe: true, hallucination_check: true, clinical_safety: true }, disclaimer: "AI-generated clinical decision support. Physician review required.", createdAt };
  if (route.startsWith("safety/")) return { synthetic_patient: { id: "SYN-2048-07", cancer: "NSCLC" }, validation: { biological: true, containsPHI: false }, module_results: { ml: "passed", dl: "passed", nlp: "partial", slm: "failed" }, blind_spots: [{ id: "BS-2048-07", difficulty: 8.7, models: ["ML", "SLM"], reason: "Cross-model calibration gap" }], createdAt };
  if (route.startsWith("tumor-board")) return { patient_state: { id: "ONC-2048" }, agent_actions: ["Patient state assembled", "Guideline evidence retrieved", "Toxicity contraindications checked", "Candidate therapies compared", "Clinical trials searched", "Safety review completed", "Treatment proposal generated"], candidate_treatments: [{ name: "Osimertinib + Savolitinib", score: .87 }], recommended_strategy: { label: "Illustrative renal-adjusted discussion option", confidence: .87 }, trial_matches: [{ count: 3, label: "potentially eligible demo trials" }], evidence: [{ id: "DEMO-GUIDELINE", source: "Configured guideline index" }], physician_review_required: true, disclaimer: "AI-generated clinical decision support. Physician review required.", createdAt };
  if (route.startsWith("analysis/")) return { patientId: "ONC-2048", toxicity: .72, progression: .81, urgency: "high", guidance: "safe", physician_review_required: true, createdAt };
  return { ok: true, mode: "demo", route, createdAt };
}

export async function GET(request: NextRequest, { params }: { params: Promise<{ route: string[] }> }) {
  const session = await getServerSession(authOptions); if (!session) return NextResponse.json({ error: "Authentication required" }, { status: 401 });
  if (!rateLimit(request)) return NextResponse.json({ error: "Rate limit exceeded" }, { status: 429 });
  const route = (await params).route.join("/");
  return NextResponse.json({ ...demo(route), demo: process.env.DEMO_MODE !== "false" });
}

export async function POST(request: NextRequest, { params }: { params: Promise<{ route: string[] }> }) {
  const session = await getServerSession(authOptions); if (!session) return NextResponse.json({ error: "Authentication required" }, { status: 401 });
  if (!rateLimit(request)) return NextResponse.json({ error: "Rate limit exceeded" }, { status: 429 });
  const route = (await params).route.join("/");
  try {
    const body: unknown = await request.json();
    const parsed = route === "slm/chat" ? slmInputSchema.parse(body) : route.startsWith("safety/") ? safetyInputSchema.parse(body) : route === "tumor-board/decision" ? decisionSchema.parse(body) : clinicalInputSchema.parse(body);
    const live = await callAIBackend(route, parsed);
    return NextResponse.json(live || { ...demo(route), demo: true, audit: { action: route, user: session.user?.email, immutable: true } });
  } catch (error) {
    const message = error instanceof Error ? error.message : "Invalid request";
    return NextResponse.json({ error: message, fallbackAvailable: true, message: "AI inference service unavailable. The dashboard remains accessible. Continue using Demo Mode or retry." }, { status: 400 });
  }
}
