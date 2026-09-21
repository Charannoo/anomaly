"use client";

import React, { useEffect, useState } from "react";
import Link from "next/link";
import { Search, Filter, RefreshCw, GitCompare } from "lucide-react";
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
    <div className="max-w-7xl mx-auto space-y-6">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4 pb-4 border-b border-border-default">
        <div>
          <h1 className="text-xl font-semibold tracking-tight text-text-primary">
            Inspection History & Logs
          </h1>
          <p className="text-xs text-text-muted mt-0.5">
            Archival record of multimodal optical and 3D surface metrology inspections.
          </p>
        </div>
        <div className="flex items-center gap-2.5">
          <Link
            href="/compare"
            className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-DEFAULT bg-bg-panel border border-border-default hover:border-accent-primary text-text-primary hover:text-accent-primary text-xs font-medium transition-colors"
          >
            <GitCompare size={14} />
            <span>Compare Inspections</span>
          </Link>
          <Link
            href="/inspect"
            className="inline-flex items-center gap-1.5 px-3.5 py-1.5 rounded-DEFAULT bg-accent-primary text-bg-app text-xs font-semibold hover:bg-accent-hover transition-colors shadow-subtle"
          >
            New Inspection
          </Link>
        </div>
      </div>

      {/* Filter Toolbar */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 bg-bg-panel border border-border-default p-3 rounded-md text-xs">
        <form onSubmit={handleSearch} className="flex-1 flex items-center gap-2 max-w-md">
          <div className="relative w-full">
            <Search size={14} className="absolute left-2.5 top-2.5 text-text-muted" />
            <input
              type="text"
              value={search}
              onChange={(e) => setSearch(e.target.value)}
              placeholder="Search by ID, sample name, or category..."
              className="w-full bg-bg-app border border-border-default rounded pl-8 pr-3 py-1.5 text-xs text-text-primary placeholder:text-text-muted outline-none focus:border-accent-primary transition-colors"
            />
          </div>
          <button
            type="submit"
            className="px-3 py-1.5 rounded bg-bg-surface hover:bg-border-emphasized border border-border-default text-text-secondary hover:text-text-primary transition-colors"
          >
            Search
          </button>
        </form>

        <div className="flex items-center gap-3">
          <div className="flex items-center gap-1.5">
            <span className="text-text-muted text-[11px]">Category:</span>
            <select
              value={category}
              onChange={(e) => setCategory(e.target.value)}
              className="bg-bg-app border border-border-default rounded px-2.5 py-1 text-xs text-text-primary outline-none"
            >
              <option value="all">All Categories</option>
              <option value="cookie">Cookie</option>
              <option value="potato">Potato</option>
              <option value="bagel">Bagel</option>
              <option value="cable_gland">Cable Gland</option>
              <option value="dowel">Dowel</option>
              <option value="peach">Peach</option>
              <option value="foam">Foam</option>
            </select>
          </div>

          <div className="flex items-center gap-1.5">
            <span className="text-text-muted text-[11px]">Status:</span>
            <select
              value={statusFilter}
              onChange={(e) => setStatusFilter(e.target.value)}
              className="bg-bg-app border border-border-default rounded px-2.5 py-1 text-xs text-text-primary outline-none"
            >
              <option value="all">All Statuses</option>
              <option value="DEFECT_DETECTED">Anomalies Only</option>
              <option value="NORMAL">Nominal Only</option>
              <option value="MANUAL_REVIEW_RECOMMENDED">Manual Review Only</option>
            </select>
          </div>

          <button
            onClick={loadData}
            className="p-1.5 rounded text-text-muted hover:text-text-primary border border-border-default bg-bg-app hover:bg-bg-surface"
            title="Refresh list"
          >
            <RefreshCw size={13} className={loading ? "animate-spin" : ""} />
          </button>
        </div>
      </div>

      {/* Table */}
      <InspectionTable inspections={inspections} />
    </div>
  );
}
