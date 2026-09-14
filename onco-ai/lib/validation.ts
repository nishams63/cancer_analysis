import { z } from "zod";

export const patientIdSchema = z.string().regex(/^ONC-\d{4,}$/);
export const clinicalInputSchema = z.object({ patientId: patientIdSchema.default("ONC-2048"), inputVersion: z.string().default("demo-v1"), payload: z.record(z.string(), z.unknown()).optional() });
export const slmInputSchema = clinicalInputSchema.extend({ message: z.string().min(1).max(4000) });
export const safetyInputSchema = z.object({ patientId: patientIdSchema.default("ONC-2048"), seed: z.coerce.number().int().nonnegative().default(2048001), cancerType: z.string().max(80).default("NSCLC") });
export const decisionSchema = z.object({ patientId: patientIdSchema, proposalId: z.string().min(1), action: z.enum(["approve", "modify", "reject", "emergency-stop"]), note: z.string().max(4000).optional() });

export function safeJson(value: unknown) { return JSON.parse(JSON.stringify(value)); }
