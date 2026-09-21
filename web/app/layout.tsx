import type { Metadata } from "next";
import "./globals.css";
import { AppShell } from "@/components/AppShell";

export const metadata: Metadata = {
  title: "PNTC Inspect — Multimodal RGB–3D Industrial Inspection",
  description: "High-precision industrial anomaly detection and 3D surface metrology platform.",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" className="dark">
      <body className="bg-bg-app text-text-primary antialiased selection:bg-accent-subtle selection:text-accent-primary">
        <AppShell>{children}</AppShell>
      </body>
    </html>
  );
}
