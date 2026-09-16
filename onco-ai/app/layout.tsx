import type { Metadata } from "next";
import { Providers } from "@/components/providers";
import "./globals.css";
import "./command-center.css";

export const metadata: Metadata = {
  title: { default: "ONCO.AI", template: "%s · ONCO.AI" },
  description: "Precision oncology intelligence platform for clinician-reviewed decision support.",
};

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en" data-scroll-behavior="smooth">
      <body>
        <Providers>{children}</Providers>
      </body>
    </html>
  );
}
