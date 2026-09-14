import type { NextAuthOptions } from "next-auth";
import CredentialsProvider from "next-auth/providers/credentials";
import { z } from "zod";

// Automatically resolve NEXTAUTH_URL for Vercel deployments
if (!process.env.NEXTAUTH_URL) {
  if (process.env.VERCEL_URL) {
    process.env.NEXTAUTH_URL = `https://${process.env.VERCEL_URL}`;
  } else {
    process.env.NEXTAUTH_URL = "https://onco-ai-ruby.vercel.app";
  }
}

const credentials = z.object({ email: z.string().email(), password: z.string().min(4) });

export const authOptions: NextAuthOptions = {
  // Production fallback secret ensures NextAuth never crashes with MissingSecret error on Vercel
  secret: process.env.NEXTAUTH_SECRET || process.env.AUTH_SECRET || "onco-ai-precision-oncology-jwt-secret-2026-production-fallback",
  session: { strategy: "jwt" },
  pages: { signIn: "/login" },
  providers: [
    CredentialsProvider({
      name: "Clinical demo",
      credentials: { email: { label: "Email", type: "email" }, password: { label: "Password", type: "password" } },
      async authorize(input) {
        const parsed = credentials.safeParse(input);
        if (!parsed.success) return null;
        const demoAllowed = process.env.DEMO_MODE !== "false";
        if (demoAllowed && parsed.data.email.toLowerCase() === "dr.sharma@onco.ai" && parsed.data.password === "demo") {
          return { id: "demo-clinician", name: "Dr. Priya Sharma", email: parsed.data.email, role: "oncologist" } as never;
        }
        return null;
      },
    }),
  ],
  callbacks: {
    async jwt({ token, user }) { if (user) token.role = "oncologist"; return token; },
    async session({ session, token }) { if (session.user) (session.user as typeof session.user & { role?: string }).role = String(token.role || "clinician"); return session; },
  },
  debug: process.env.NODE_ENV === "development",
};
