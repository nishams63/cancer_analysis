import "server-only";
import { z } from "zod";

const backendEnvelope = z.object({ ok: z.boolean().optional() }).passthrough();

export async function callAIBackend(path: string, body: unknown) {
  if (process.env.DEMO_MODE !== "false" || !process.env.AI_BACKEND_URL) return null;
  const response = await fetch(`${process.env.AI_BACKEND_URL.replace(/\/$/, "")}/${path.replace(/^\//, "")}`, {
    method: "POST",
    headers: { "content-type": "application/json", authorization: `Bearer ${process.env.AI_BACKEND_API_KEY || ""}` },
    body: JSON.stringify(body),
    cache: "no-store",
    signal: AbortSignal.timeout(30_000),
  });
  if (!response.ok) throw new Error(`AI backend unavailable (${response.status})`);
  return backendEnvelope.parse(await response.json());
}
