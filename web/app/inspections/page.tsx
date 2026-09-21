"use client";

import React, { useEffect, useState } from "react";
import Link from "next/link";
import { Search, RefreshCw, GitCompare, Plus } from "lucide-react";
import { fetchInspections, InspectionListItem } from "@/lib/api";
import { InspectionTable } from "@/components/InspectionTable";

export default function InspectionsHistoryPage() {
  const [inspections, setInspections] = useState<InspectionListItem[]>([]);
  const [loading, setLoading] = useState(true);
  const [search, setSearch] = useState("");
  const [category, setCategory] = useState("all");
  const [statusFilter, setStatusFilter] = useState("all");

  const loadData = () => {
    setLoading(true);
    fetchInspections({
      search: search || undefined,
      category: category !== "all" ? category : undefined,
      status: statusFilter !== "all" ? statusFilter : undefined,
      limit: 100,
    })
      .then((data) => setInspections(data.items))
      .catch(console.error)
      .finally(() => setLoading(false));
  };

  useEffect(() => {
    loadData();
  }, [category, statusFilter]);

  const handleSearch = (e: React.FormEvent) => {
    e.preventDefault();
    loadData();
  };

  return (
    <div className="max-w-[1500px] mx-auto space-y-8">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-end sm:justify-between gap-4 pb-6 border-b border-white/[0.05]">
        <div>
          <h1 className="text-2xl sm:text-[28px] font-semibold tracking-tight text-text-primary">
            Inspections
          </h1>
          <p className="text-sm text-text-muted mt-1">
            Archival record of multimodal optical and 3D surface metrology inspections.
          </p>
        </div>
        <div className="flex items-center gap-3">
          <Link
            href="/compare"
            className="inline-flex items-center gap-1.5 px-3.5 py-1.5 rounded bg-bg-surface hover:bg-bg-elevated border border-white/[0.08] text-text-secondary hover:text-text-primary text-xs font-medium transition-colors"
          >
            <GitCompare size={14} />
            <span>Compare</span>
          </Link>
          <Link
            href="/inspect"
            className="inline-flex items-center gap-1.5 px-4 py-1.5 rounded bg-accent-primary hover:bg-accent-hover text-bg-app text-xs font-semibold transition-colors shadow-sm"
          >
            <Plus size={14} />
            <span>New inspection</span>
          </Link>
        </div>
      </div>

      {/* Filter Toolbar - Clean, Non-Boxed */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 text-xs">
        <form onSubmit={handleSearch} className="flex-1 flex items-center gap-2 max-w-md">
          <div className="relative w-full">
            <Search size={14} className="absolute left-3 top-2.5 text-text-muted" />
            <input
              type="text"
              value={search}
              onChange={(e) => setSearch(e.target.value)}
              placeholder="Search by ID, sample, or category..."
              className="w-full bg-bg-surface border border-white/[0.08] rounded pl-9 pr-3 py-1.5 text-xs text-text-primary placeholder:text-text-muted outline-none focus:border-accent-primary transition-colors"
            />
          </div>
          <button
            type="submit"
            className="px-3.5 py-1.5 rounded bg-bg-surface hover:bg-bg-elevated border border-white/[0.08] text-text-secondary hover:text-text-primary transition-colors font-medium"
          >
            Search
          </button>
        </form>

        <div className="flex items-center gap-3">
          <div className="flex items-center gap-2">
            <span className="text-text-muted text-xs">Category</span>
            <select
              value={category}
              onChange={(e) => setCategory(e.target.value)}
              className="bg-bg-surface border border-white/[0.08] rounded px-3 py-1.5 text-xs text-text-primary outline-none focus:border-accent-primary cursor-pointer"
            >
              <option value="all">All categories</option>
              <option value="cookie">Cookie</option>
              <option value="potato">Potato</option>
              <option value="bagel">Bagel</option>
              <option value="cable_gland">Cable gland</option>
              <option value="dowel">Dowel</option>
              <option value="peach">Peach</option>
              <option value="foam">Foam</option>
            </select>
          </div>

          <div className="flex items-center gap-2">
            <span className="text-text-muted text-xs">Status</span>
            <select
              value={statusFilter}
              onChange={(e) => setStatusFilter(e.target.value)}
              className="bg-bg-surface border border-white/[0.08] rounded px-3 py-1.5 text-xs text-text-primary outline-none focus:border-accent-primary cursor-pointer"
            >
              <option value="all">All statuses</option>
              <option value="DEFECT_DETECTED">Anomalies</option>
              <option value="NORMAL">Normal</option>
              <option value="MANUAL_REVIEW_RECOMMENDED">Manual review</option>
            </select>
          </div>

          <button
            onClick={loadData}
            className="p-2 rounded text-text-muted hover:text-text-primary border border-white/[0.08] bg-bg-surface hover:bg-bg-elevated transition-colors"
            title="Refresh list"
          >
            <RefreshCw size={13} className={loading ? "animate-spin" : ""} />
          </button>
        </div>
      </div>

      {/* Table with Breathing Room */}
      <div className="pt-2">
        <InspectionTable inspections={inspections} />
      </div>
    </div>
  );
}
