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
  Sparkles,
  BookOpen,
  Settings,
  ChevronLeft,
  ChevronRight,
} from "lucide-react";
import { cn } from "@/lib/utils";
import { fetchProviderStatus, fetchSystemStatus } from "@/lib/api";

const PRIMARY_NAV = [
  { label: "Overview", href: "/dashboard", icon: LayoutDashboard },
  { label: "New Inspection", href: "/inspect", icon: PlusCircle },
  { label: "Inspections", href: "/inspections", icon: History },
  { label: "Compare", href: "/compare", icon: GitCompare },
  { label: "Analytics", href: "/analytics", icon: BarChart3 },
  { label: "Demo Mode", href: "/demo", icon: Sparkles },
];

const SECONDARY_NAV = [
  { label: "About", href: "/about", icon: BookOpen },
  { label: "Settings", href: "/settings", icon: Settings },
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

  const allNav = [...PRIMARY_NAV, ...SECONDARY_NAV];
  const activeNav = allNav.find(
    (n) => pathname === n.href || (n.href !== "/dashboard" && pathname.startsWith(n.href))
  );

  let displayTitle = title || activeNav?.label || "Inspection System";
  let displayBreadcrumb = breadcrumb;

  // Contextual breadcrumb resolution
  if (!displayBreadcrumb) {
    if (pathname.startsWith("/inspect/INSP-") || (pathname.startsWith("/inspect/") && pathname !== "/inspect")) {
      displayTitle = "Inspection";
      const parts = pathname.split("/");
      displayBreadcrumb = parts[parts.length - 1];
    }
  }

  return (
    <div className="flex h-screen w-screen overflow-hidden bg-[#0B0D10] text-[#F3F5F7] font-sans">
      {/* Sidebar */}
      <aside
        className={cn(
          "flex flex-col border-r border-white/[0.06] bg-[#0E1115] transition-all duration-200 z-30 shrink-0",
          collapsed ? "w-[56px]" : "w-[204px]"
        )}
      >
        {/* Wordmark Header */}
        <div className="flex h-14 items-center justify-between border-b border-white/[0.06] px-3.5">
          {!collapsed ? (
            <div className="flex flex-col">
              <span className="text-[13px] font-semibold tracking-tight text-[#F3F5F7]">
                PNTC Inspect
              </span>
              <span className="text-[10.5px] text-[#6F7884]">Industrial RGB–3D</span>
            </div>
          ) : (
            <div className="mx-auto flex h-6 w-6 items-center justify-center rounded bg-white/[0.04] text-[#5BB8C4] font-semibold text-xs">
              P
            </div>
          )}
          <button
            onClick={() => setCollapsed(!collapsed)}
            className="p-1 rounded text-[#6F7884] hover:text-[#F3F5F7] hover:bg-white/[0.04] transition-colors"
            title={collapsed ? "Expand sidebar" : "Collapse sidebar"}
          >
            {collapsed ? <ChevronRight size={13} /> : <ChevronLeft size={13} />}
          </button>
        </div>

        {/* Navigation Sections */}
        <nav className="flex-1 space-y-0.5 p-2 overflow-y-auto">
          {PRIMARY_NAV.map((item) => {
            const Icon = item.icon;
            const isActive = pathname === item.href || (item.href !== "/dashboard" && pathname.startsWith(item.href));
            return (
              <Link
                key={item.href}
                href={item.href}
                className={cn(
                  "flex items-center gap-2.5 px-2.5 py-1.5 rounded text-[13px] transition-colors",
                  isActive
                    ? "bg-[#171C22] text-[#F3F5F7] font-medium"
                    : "text-[#A7AFBA] hover:text-[#F3F5F7] hover:bg-white/[0.03]"
                )}
                title={collapsed ? item.label : undefined}
              >
                <Icon size={15} className={cn("shrink-0", isActive ? "text-[#5BB8C4]" : "text-[#6F7884]")} />
                {!collapsed && <span>{item.label}</span>}
              </Link>
            );
          })}

          <div className="my-2 border-t border-white/[0.05] mx-1" />

          {SECONDARY_NAV.map((item) => {
            const Icon = item.icon;
            const isActive = pathname === item.href || pathname.startsWith(item.href);
            return (
              <Link
                key={item.href}
                href={item.href}
                className={cn(
                  "flex items-center gap-2.5 px-2.5 py-1.5 rounded text-[13px] transition-colors",
                  isActive
                    ? "bg-[#171C22] text-[#F3F5F7] font-medium"
                    : "text-[#A7AFBA] hover:text-[#F3F5F7] hover:bg-white/[0.03]"
                )}
                title={collapsed ? item.label : undefined}
              >
                <Icon size={15} className={cn("shrink-0", isActive ? "text-[#5BB8C4]" : "text-[#6F7884]")} />
                {!collapsed && <span>{item.label}</span>}
              </Link>
            );
          })}
        </nav>

        {/* Quiet Minimal Status Footer */}
        <div className="border-t border-white/[0.06] p-3 text-xs bg-[#0E1115]">
          {!collapsed ? (
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-1.5">
                <span className="w-1.5 h-1.5 rounded-full bg-[#55B98A]" />
                <span className="text-[11px] font-medium text-[#A7AFBA]">System Ready</span>
              </div>
              <span className="text-[10.5px] font-mono text-[#6F7884]">v5.4</span>
            </div>
          ) : (
            <div className="flex justify-center" title="System Ready · v5.4">
              <span className="w-1.5 h-1.5 rounded-full bg-[#55B98A]" />
            </div>
          )}
        </div>
      </aside>

      {/* Main Content Workspace */}
      <div className="flex flex-1 flex-col overflow-hidden">
        {/* Quiet Minimal Top Bar */}
        <header className="flex h-14 items-center justify-between border-b border-white/[0.06] bg-[#0E1115] px-8 shrink-0">
          <div className="flex items-center gap-2">
            <span className="text-[13px] font-semibold text-[#F3F5F7]">
              {displayTitle}
            </span>
            {displayBreadcrumb && (
              <>
                <span className="text-xs text-[#424852]">/</span>
                <span className="text-xs text-[#A7AFBA]">
                  {displayBreadcrumb}
                </span>
              </>
            )}
          </div>

          <div className="flex items-center gap-3">
            <div className="flex items-center gap-1.5 text-[11px] text-[#A7AFBA]">
              <span className="w-1.5 h-1.5 rounded-full bg-[#55B98A]" />
              <span>System Ready</span>
            </div>
            <div className="flex items-center gap-1 px-2 py-0.5 rounded bg-white/[0.03] border border-white/[0.06] text-[11px] text-[#A7AFBA]">
              <span className="capitalize">{providerInfo.provider}</span>
            </div>
          </div>
        </header>

        {/* Main Workspace Body */}
        <main className="flex-1 overflow-y-auto bg-[#0B0D10] px-8 py-7 md:px-10 md:py-8">
          {children}
        </main>
      </div>
    </div>
  );
};

export default AppShell;
