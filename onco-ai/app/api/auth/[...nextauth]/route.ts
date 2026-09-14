import NextAuth from "next-auth";
import { authOptions } from "@/lib/auth/options";

if (!process.env.NEXTAUTH_URL) {
  if (process.env.VERCEL_URL) {
    process.env.NEXTAUTH_URL = `https://${process.env.VERCEL_URL}`;
  } else {
    process.env.NEXTAUTH_URL = "https://onco-ai-ruby.vercel.app";
  }
}

const handler = NextAuth(authOptions);
export { handler as GET, handler as POST };
