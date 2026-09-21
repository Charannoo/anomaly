"use client";

import React, { useState, useEffect } from "react";
import Link from "next/link";
import { usePathname } from "next/navigation";
import {
  LayoutDashboard,
  PlusCircle,
  History,
  GitCompare,
  BarChart3,
  Cpu,
  BookOpen,
  Settings,
  Sparkles,
  ChevronLeft,
  ChevronRight,
  ShieldCheck,
  CheckCircle2,
} from "lucide-react";
import { cn } from "@/lib/utils";
import { fetchProviderStatus, fetchSystemStatus } from "@/lib/api";

const NAV_ITEMS = [
  { label: "Overview", href: "/dashboard", icon: LayoutDashboard, subtitle: "Operational Metrology" },
  { label: "New Inspection", href: "/inspect", icon: PlusCircle, subtitle: "Dual-Modality Observation" },
  { label: "Inspections", href: "/inspections", icon: History, subtitle: "Quality Archive" },
  { label: "Compare", href: "/compare", icon: GitCompare, subtitle: "Side-by-Side Analysis" },
  { label: "Analytics", href: "/analytics", icon: BarChart3, subtitle: "Operational Metrics" },
  { label: "Model Explorer", href: "/model", icon: Cpu, subtitle: "Architecture & Verification" },
  { label: "Research", href: "/research", icon: BookOpen, subtitle: "Scientific Formulation" },
  { label: "Demo Mode", href: "/demo", icon: Sparkles, subtitle: "Faculty Demonstration" },
  { label: "Settings", href: "/settings", icon: Settings, subtitle: "Configuration & Providers" },
];

interface AppShellProps {
  children: React.ReactNode;
  title?: string;
  breadcrumb?: string;
}

export const AppShell: React.FC<AppShellProps> = ({ children, title, breadcrumb }) => {
  const pathname = usePathname();
  const [collapsed, setCollapsed] = useState(false);
  const [providerInfo, setProviderInfo] = useState({ provider: "gemini", configured: true });
  const [systemInfo, setSystemInfo] = useState({ status: "OPERATIONAL", version: "5.4.0" });

  useEffect(() => {
    fetchProviderStatus()
      .then((data) => setProviderInfo(data))
      .catch(() => {});
    fetchSystemStatus()
      .then((data) => setSystemInfo(data))
      .catch(() => {});
  }, []);

  return (
    <div className="flex h-screen w-screen overflow-hidden bg-bg-app text-text-primary">
      {/* Sidebar */}
      <aside
        className={cn(
          "flex flex-col border-r border-border-default bg-bg-subtle transition-all duration-200 z-30",
          collapsed ? "w-[60px]" : "w-[230px]"
        )}
      >
        {/* Brand Header */}
        <div className="flex h-14 items-center justify-between border-b border-border-default px-3.5">
          {!collapsed && (
            <div className="flex flex-col">
              <span className="text-[13px] font-semibold tracking-tight text-text-primary flex items-center gap-1.5">
                <span className="w-2 h-2 rounded-full bg-accent-primary" />
                PNTC Inspect
              </span>
              <span className="text-[10px] text-text-muted">RGB–3D Metrology</span>
            </div>
          )}
          {collapsed && (
            <div className="mx-auto flex h-7 w-7 items-center justify-center rounded bg-accent-subtle text-accent-primary font-bold text-xs">
              P
            </div>
          )}
          <button
            onClick={() => setCollapsed(!collapsed)}
            className="p-1 rounded text-text-muted hover:text-text-primary hover:bg-bg-surface transition-colors"
            title={collapsed ? "Expand sidebar" : "Collapse sidebar"}
          >
            {collapsed ? <ChevronRight size={14} /> : <ChevronLeft size={14} />}
          </button>
        </div>

        {/* Navigation */}
        <nav className="flex-1 space-y-1 p-2 overflow-y-auto">
          {NAV_ITEMS.map((item) => {
            const Icon = item.icon;
            const isActive = pathname === item.href || (item.href !== "/dashboard" && pathname.startsWith(item.href));
            return (
              <Link
                key={item.href}
                href={item.href}
                className={cn(
                  "flex items-center gap-2.5 px-2.5 py-1.5 rounded-DEFAULT text-[13px] font-medium transition-colors",
                  isActive
                    ? "bg-bg-active text-accent-primary border-l-2 border-accent-primary font-semibold"
                    : "text-text-secondary hover:text-text-primary hover:bg-bg-surface"
                )}
                title={collapsed ? item.label : undefined}
              >
                <Icon size={16} className="flex-shrink-0" />
                {!collapsed && <span>{item.label}</span>}
              </Link>
            );
          })}
        </nav>

        {/* Sidebar Footer */}
        <div className="border-t border-border-default p-2.5 text-[11px] bg-bg-app">
          {!collapsed ? (
            <div className="space-y-1 text-text-muted">
              <div className="flex items-center justify-between">
                <span className="text-text-secondary">Core Detector</span>
                <span className="font-mono text-[10px] text-status-normal font-medium">PNTC Frozen</span>
              </div>
              <div className="flex items-center justify-between">
                <span>Canonical Tag</span>
                <span className="font-mono text-[10px]">h5d-verified</span>
              </div>
              <div className="flex items-center justify-between">
                <span>Version</span>
                <span className="font-mono text-[10px]">v{systemInfo.version}</span>
              </div>
            </div>
          ) : (
            <div className="flex justify-center" title="PNTC Frozen: h5d-pntc-verified">
              <ShieldCheck size={16} className="text-status-normal" />
            </div>
          )}
        </div>
      </aside>

      {/* Main Container */}
      <div className="flex flex-1 flex-col overflow-hidden">
        {/* Top Bar */}
        {(() => {
          const activeNav = NAV_ITEMS.find((n) => pathname === n.href || (n.href !== "/dashboard" && pathname.startsWith(n.href)));
          const displayTitle = title || activeNav?.label || "Inspection System";
          const displayBreadcrumb = breadcrumb || activeNav?.subtitle || "Multimodal RGB–3D";
          return (
            <header className="flex h-14 items-center justify-between border-b border-border-default bg-bg-subtle px-6">
              <div className="flex items-center gap-3">
                <span className="text-sm font-semibold text-text-primary">
                  {displayTitle}
                </span>
                <span className="text-xs text-text-muted">/</span>
                <span className="text-xs text-text-secondary font-mono">
                  {displayBreadcrumb}
                </span>
              </div>

              <div className="flex items-center gap-4">
                {/* Frozen Benchmark Badge */}
                <div className="hidden md:flex items-center gap-1.5 px-2.5 py-1 rounded bg-bg-panel border border-border-default text-[11px] font-mono text-text-muted">
                  <span>CANONICAL:</span>
                  <span className="text-status-normal font-semibold">I-AUROC 0.96541</span>
                </div>

                {/* Provider Pill */}
                <div className="flex items-center gap-1.5 px-2.5 py-1 rounded bg-accent-subtle border border-border-subtle text-[11px] font-mono text-accent-primary">
                  <span className="w-1.5 h-1.5 rounded-full bg-accent-primary animate-pulse" />
                  <span className="capitalize">{providerInfo.provider}</span>
                  <span className="text-text-muted">Active</span>
                </div>

                {/* System Status Pill */}
                <div className="flex items-center gap-1.5 text-[11px] font-mono text-status-normal">
                  <CheckCircle2 size={12} />
                  <span>{systemInfo.status}</span>
                </div>
              </div>
            </header>
          );
        })()}

        {/* Workspace Content */}
        <main className="flex-1 overflow-y-auto bg-bg-app p-6">
          {children}
        </main>
      </div>
    </div>
  );
};

export default AppShell;

