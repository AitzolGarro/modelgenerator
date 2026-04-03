import type { Job, JobListResponse, JobCreatePayload, HealthResponse } from "@/types/job";

const API_BASE = "/api/v1";

async function fetchAPI<T>(path: string, options?: RequestInit): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, {
    headers: {
      "Content-Type": "application/json",
      ...options?.headers,
    },
    ...options,
  });

  if (!res.ok) {
    const error = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(error.detail || `API error: ${res.status}`);
  }

  if (res.status === 204) return undefined as T;
  return res.json();
}

// --- Jobs ---

export async function createJob(payload: JobCreatePayload): Promise<Job> {
  return fetchAPI<Job>("/jobs", {
    method: "POST",
    body: JSON.stringify(payload),
  });
}

export async function uploadAndCreateJob(
  file: File,
  jobType: "animate" | "refine" | "skin",
  prompt: string,
  negativePrompt?: string,
): Promise<Job> {
  const formData = new FormData();
  formData.append("file", file);
  formData.append("job_type", jobType);
  formData.append("prompt", prompt);
  if (negativePrompt) formData.append("negative_prompt", negativePrompt);

  const res = await fetch(`${API_BASE}/jobs/upload`, {
    method: "POST",
    body: formData,
    // Don't set Content-Type header — browser sets it with boundary for multipart
  });

  if (!res.ok) {
    const error = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(error.detail || `API error: ${res.status}`);
  }

  return res.json();
}

export async function listJobs(
  page = 1,
  pageSize = 20,
  status?: string,
  jobType?: string,
): Promise<JobListResponse> {
  const params = new URLSearchParams({
    page: String(page),
    page_size: String(pageSize),
  });
  if (status) params.set("status", status);
  if (jobType) params.set("job_type", jobType);
  return fetchAPI<JobListResponse>(`/jobs?${params}`);
}

export async function getJob(id: number): Promise<Job> {
  return fetchAPI<Job>(`/jobs/${id}`);
}

export async function deleteJob(id: number): Promise<void> {
  return fetchAPI<void>(`/jobs/${id}`, { method: "DELETE" });
}

export async function retryJob(id: number): Promise<Job> {
  return fetchAPI<Job>(`/jobs/${id}/retry`, { method: "POST" });
}

// --- Health ---

export async function getHealth(): Promise<HealthResponse> {
  return fetchAPI<HealthResponse>("/health");
}

export function getFileUrl(relativePath: string): string {
  return `${API_BASE}/files/${relativePath}`;
}
