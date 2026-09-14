"use client";

import { useEffect, useState } from "react";
import { signIn, useSession } from "next-auth/react";
import { useRouter } from "next/navigation";
import { Activity, ArrowRight, BrainCircuit, Check, Dna, Eye, EyeOff, ShieldCheck, Stethoscope } from "lucide-react";

export default function LoginPage() {
  const router = useRouter();
  const { status } = useSession();
  const [show, setShow] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");

  useEffect(() => {
    if (status === "authenticated") {
      router.replace("/overview");
    }
  }, [status, router]);

  async function submit(e: React.FormEvent<HTMLFormElement>) {
    e.preventDefault();
    setBusy(true);
    setError("");
    try {
      const data = new FormData(e.currentTarget);
      const email = data.get("email") as string;
      const password = data.get("password") as string;
      const result = await signIn("credentials", {
        email,
        password,
        redirect: false,
        callbackUrl: "/overview",
      });
      setBusy(false);
      if (result?.ok) {
        router.push("/overview");
        router.refresh();
      } else {
        if (result?.error === "Configuration") {
          setError("Authentication configuration error. Please verify server environment variables.");
        } else {
          setError("Invalid credentials. Use the demo credentials shown below.");
        }
      }
    } catch (err) {
      setBusy(false);
      setError("An unexpected authentication error occurred. Please try again.");
    }
  }

  return (
    <main className="login-page">
      <section className="login-hero" aria-label="ONCO.AI precision oncology introduction">
        <div className="login-orb orb-one" /><div className="login-orb orb-two" />
        <div className="dna-art" aria-hidden="true"><Dna /></div>
        <div className="brand-lockup"><span className="brand-mark"><Dna /></span><div><strong>ONCO.AI</strong><small>Precision Oncology Intelligence</small></div></div>
        <div className="hero-copy">
          <p className="eyebrow">MULTI-MODAL CLINICAL INTELLIGENCE</p>
          <h1>From Data to Decisions<br />For a Healthier Tomorrow</h1>
          <p>AI-powered multi-modal intelligence for personalized cancer care.</p>
          <div className="concepts">
            <span><Activity />Predict</span><span><BrainCircuit />Understand</span><span><Stethoscope />Plan</span><span><ShieldCheck />Protect</span>
          </div>
        </div>
        <blockquote>“Smarter insights.<br />Brighter tomorrows.”</blockquote>
        <small className="hero-foot">Trusted AI. Human-Centered Care.</small>
      </section>
      <section className="login-panel">
        <form className="login-card" onSubmit={submit}>
          <div className="mobile-brand"><span className="brand-mark"><Dna /></span> ONCO.AI</div>
          <p className="eyebrow">CLINICIAN PORTAL</p><h2>Welcome back</h2><p>Sign in to continue to the oncology command center.</p>
          <label>Email address<input name="email" type="email" defaultValue="dr.sharma@onco.ai" autoComplete="email" required /></label>
          <label>Password<div className="password-wrap"><input name="password" type={show ? "text" : "password"} defaultValue="demo" autoComplete="current-password" required /><button type="button" aria-label={show ? "Hide password" : "Show password"} onClick={() => setShow(!show)}>{show ? <EyeOff /> : <Eye />}</button></div></label>
          <div className="login-row"><label className="check"><input type="checkbox" defaultChecked /> Remember me</label><button type="button" className="link-button">Forgot password?</button></div>
          {error && <div className="form-error" role="alert">{error}</div>}
          <button className="primary wide" disabled={busy}>{busy ? "Signing in…" : <>Sign in <ArrowRight /></>}</button>
          <div className="demo-callout"><Check /> Demo access is prefilled: <b>dr.sharma@onco.ai</b> / <b>demo</b></div>
          <div className="divider"><span>or continue with</span></div>
          <div className="social-row"><button type="button">Google</button><button type="button">Microsoft</button></div>
          <p className="create-copy">New to ONCO.AI? <button type="button" className="link-button">Create an account</button></p>
          <p className="clinical-disclaimer">AI-generated clinical decision support. Physician review required.</p>
        </form>
      </section>
    </main>
  );
}
