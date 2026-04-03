export type JobStatus =
  | "pending"
  | "generating_image"
  | "image_ready"
  | "generating_model"
  | "model_ready"
  | "texturing"
  | "exporting"
  | "completed"
  | "failed";

export interface Job {
  id: number;
  prompt: string;
  negative_prompt: string | null;
  status: JobStatus;
  image_path: string | null;
  model_path: string | null;
  textured_model_path: string | null;
  export_path: string | null;
  image_url: string | null;
  model_url: string | null;
  export_url: string | null;
  error_message: string | null;
  num_steps: number;
  guidance_scale: number;
  seed: number | null;
  retry_count: number;
  created_at: string;
  updated_at: string;
  completed_at: string | null;
}

export interface JobListResponse {
  jobs: Job[];
  total: number;
  page: number;
  page_size: number;
}

export interface JobCreatePayload {
  prompt: string;
  negative_prompt?: string;
  num_steps?: number;
  guidance_scale?: number;
  seed?: number;
}

export interface HealthResponse {
  status: string;
  version: string;
  gpu_available: boolean;
  gpu_name: string | null;
}

// Status display helpers
export const STATUS_LABELS: Record<JobStatus, string> = {
  pending: "En cola",
  generating_image: "Generando imagen",
  image_ready: "Imagen lista",
  generating_model: "Generando modelo 3D",
  model_ready: "Modelo listo",
  texturing: "Aplicando textura",
  exporting: "Exportando",
  completed: "Completado",
  failed: "Error",
};

export const STATUS_COLORS: Record<JobStatus, string> = {
  pending: "bg-gray-400",
  generating_image: "bg-yellow-400 animate-pulse",
  image_ready: "bg-yellow-500",
  generating_model: "bg-blue-400 animate-pulse",
  model_ready: "bg-blue-500",
  texturing: "bg-purple-400 animate-pulse",
  exporting: "bg-indigo-400 animate-pulse",
  completed: "bg-green-500",
  failed: "bg-red-500",
};

export function isProcessing(status: JobStatus): boolean {
  return ![
    "pending",
    "completed",
    "failed",
    "image_ready",
    "model_ready",
  ].includes(status);
}
