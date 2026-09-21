/** API Client for PNTC Inspect Backend */

const API_BASE = "/api";

export interface InspectionListItem {
  id: string;
  sample_id: string;
  category: string;
  status: string;
  decision: string;
  anomaly_score: number;
  operating_threshold: number;
  num_defects: number;
  decision_certainty: string;
  measurement_reliability: string;
  manual_review_recommended: boolean;
  execution_time_ms: number;
  created_at: string;
  artifacts_dir: string;
}

export interface InspectionDetail extends InspectionListItem {
  report: any;
}

export interface AnalyticsData {
  total_inspections: number;
  anomalies: number;
  normal: number;
  manual_review: number;
  average_anomaly_score: number;
  manual_review_rate: number;
  by_category: { category: string; count: number }[];
  by_status: { status: string; count: number }[];
}

export interface DemoCase {
  id: string;
  name: string;
  category: string;
  type: string;
  expected_decision: string;
  expected_score: number;
  highlight: string;
  evidence: string;
  thumbnail_url: string;
}

export async function fetchInspections(params?: {
  category?: string;
  status?: string;
  search?: string;
  limit?: number;
  offset?: number;
}): Promise<{ total: number; items: InspectionListItem[] }> {
  const query = new URLSearchParams();
  if (params?.category) query.set("category", params.category);
  if (params?.status) query.set("status", params.status);
  if (params?.search) query.set("search", params.search);
  if (params?.limit) query.set("limit", String(params.limit));
  if (params?.offset) query.set("offset", String(params.offset));

  const res = await fetch(`${API_BASE}/inspections?${query.toString()}`);
  if (!res.ok) throw new Error("Failed to load inspections");
  return res.json();
}

export async function fetchInspection(id: string): Promise<InspectionDetail> {
  const res = await fetch(`${API_BASE}/inspections/${id}`);
  if (!res.ok) throw new Error(`Inspection ${id} not found`);
  return res.json();
}

export async function fetchAnalytics(): Promise<AnalyticsData> {
  const res = await fetch(`${API_BASE}/analytics`);
  if (!res.ok) throw new Error("Failed to load analytics");
  return res.json();
}

export async function fetchDemoCases(): Promise<DemoCase[]> {
  const res = await fetch(`${API_BASE}/demo/cases`);
  if (!res.ok) throw new Error("Failed to load demo cases");
  return res.json();
}

export async function runInspection(payload: {
  demo_case_id?: string;
  sample_id?: string;
  category?: string;
  sample_condition?: string;
  generate_3d?: boolean;
  generate_assistant_summary?: boolean;
  response_mode?: string;
}): Promise<{ inspection_id: string; sample_id: string; status: string; decision: string; anomaly_score?: number }> {
  const res = await fetch(`${API_BASE}/inspections/run`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  if (!res.ok) throw new Error("Inspection execution failed");
  return res.json();
}

export async function fetchModelInfo(): Promise<any> {
  const res = await fetch(`${API_BASE}/model/info`);
  if (!res.ok) throw new Error("Failed to load model info");
  return res.json();
}

export async function fetchResearchAblations(): Promise<any> {
  const res = await fetch(`${API_BASE}/research/ablations`);
  if (!res.ok) throw new Error("Failed to load research ablations");
  return res.json();
}

export async function fetchAssistantExplanation(
  sampleId: string,
  mode: string = "TECHNICAL",
  forceRefresh: boolean = false
): Promise<any> {
  const res = await fetch(`${API_BASE}/assistant/explain`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ sample_id: sampleId, response_mode: mode, force_refresh: forceRefresh }),
  });
  if (!res.ok) throw new Error("Assistant explanation failed");
  return res.json();
}

export async function sendAssistantChat(
  sampleId: string,
  message: string,
  conversationId?: string,
  mode: string = "TECHNICAL",
  defectId?: number
): Promise<any> {
  const res = await fetch(`${API_BASE}/assistant/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      sample_id: sampleId,
      message,
      conversation_id: conversationId,
      response_mode: mode,
      defect_id: defectId,
    }),
  });
  if (!res.ok) throw new Error("Assistant chat failed");
  return res.json();
}

export async function fetchSystemStatus(): Promise<any> {
  const res = await fetch(`${API_BASE}/system/status`);
  if (!res.ok) throw new Error("System status unavailable");
  return res.json();
}

export async function fetchProviderStatus(): Promise<any> {
  const res = await fetch(`${API_BASE}/assistant/provider`);
  if (!res.ok) throw new Error("Provider status unavailable");
  return res.json();
}
