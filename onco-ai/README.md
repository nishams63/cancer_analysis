# ONCO.AI — Precision Oncology Intelligence Platform

Production-oriented Next.js application shell for a six-stage, clinician-reviewed oncology intelligence workflow. The bundled demo uses clearly synthetic data for patient `ONC-2048`; it does not provide diagnosis or treatment.

## Quick start

```bash
npm install
cp .env.example .env.local
npm run dev
```

Open `http://localhost:3000` and use:

- Email: `dr.sharma@onco.ai`
- Password: `demo`

The credential is enabled only when `DEMO_MODE` is not set to `false`.

## Architecture

- Next.js App Router, React, TypeScript, Tailwind CSS
- Auth.js/NextAuth credentials flow for the self-contained demo
- TanStack Query provider, Zustand UI state, React Hook Form/Zod-ready validation
- Recharts visualizations and Framer Motion transitions
- PostgreSQL schema via Prisma, with append-only prediction and proposal versions
- Authenticated, rate-limited internal API gateway under `/api/*`
- Server-only external AI client; backend keys are never sent to the browser
- S3/R2 and Redis environment placeholders for production services

All AI routes return realistic, labeled demo output while `DEMO_MODE=true`. When `DEMO_MODE=false`, the server forwards validated requests to `AI_BACKEND_URL`. Failed inference never crashes the dashboard and returns a safe fallback message.

## Doc AI portrait

The UI deliberately does not invent a substitute person. Add the original user-approved transparent portrait at:

`public/doc-ai/doc-ai.png`

The assistant currently uses a neutral clinical-assistant glyph and retains the full panel, state, queue-ready provider, contextual messages, typing state, voice/microphone placeholders, and page-aware suggestions.

## Database

Create a PostgreSQL database, set `DATABASE_URL`, then run:

```bash
npm run prisma:generate
npx prisma migrate dev --name init
npm run db:seed
```

Prediction tables are append-only by design: each call creates a new record with its own model version, input version, confidence, metadata, and timestamp.

## Deploy to Vercel

1. Push `onco-ai` to GitHub.
2. Import it into Vercel as a Next.js project.
3. Provision PostgreSQL and configure the variables in `.env.example`.
4. Run Prisma migrations against the production database.
5. Keep `DEMO_MODE=true` until the external FastAPI service is ready.
6. Set `AI_BACKEND_URL` and `AI_BACKEND_API_KEY`, then switch `DEMO_MODE=false`.

Recommended production additions: Redis-backed distributed rate limiting, database-backed Auth.js sessions, signed R2 uploads, immutable audit retention, and organization-specific guideline licensing.

## Clinical safety

- Every Stage 4 and Stage 6 output carries a physician-review disclaimer.
- Stage 6 exposes actions, evidence, observations, and summaries—not hidden chain-of-thought.
- Approve, modify, reject, and emergency-stop interactions are visibly distinct and audit-ready.
- Demo responses are never presented as real inference.

This software is a clinical decision-support prototype, not a medical device.
