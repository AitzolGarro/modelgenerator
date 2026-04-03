export type JobType = "generate" | "animate" | "refine" | "scene";

export type JobStatus =
  | "pending"
  | "generating_image"
  | "image_ready"
  | "generating_model"
  | "model_ready"
  | "texturing"
  | "rigging"
  | "animating"
  | "refining"
  | "generating_scene"
  | "compositing"
  | "exporting"
  | "completed"
  | "failed";

export interface Job {
  id: number;
  job_type: JobType;
  prompt: string;
  negative_prompt: string | null;
  status: JobStatus;
  input_file_path: string | null;
  image_path: string | null;
  model_path: string | null;
  textured_model_path: string | null;
  export_path: string | null;
  input_file_url: string | null;
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
  job_type?: JobType;
  prompt: string;
  negative_prompt?: string;
  num_steps?: number;
  guidance_scale?: number;
  seed?: number;
  source_job_id?: number;
}

export interface HealthResponse {
  status: string;
  version: string;
  gpu_available: boolean;
  gpu_name: string | null;
}

export const JOB_TYPE_LABELS: Record<JobType, string> = {
  generate: "Generar 3D",
  animate: "Animar",
  refine: "Mejorar",
  scene: "Escenario",
};

export const JOB_TYPE_ICONS: Record<JobType, string> = {
  generate: "cube",
  animate: "play",
  refine: "sparkles",
  scene: "mountain",
};

export const STATUS_LABELS: Record<JobStatus, string> = {
  pending: "En cola",
  generating_image: "Generando imagen",
  image_ready: "Imagen lista",
  generating_model: "Generando modelo 3D",
  model_ready: "Modelo listo",
  texturing: "Aplicando textura",
  rigging: "Rigging",
  animating: "Animando",
  refining: "Mejorando detalle",
  generating_scene: "Generando escenario",
  compositing: "Componiendo escena",
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
  rigging: "bg-orange-400 animate-pulse",
  animating: "bg-orange-500 animate-pulse",
  refining: "bg-teal-400 animate-pulse",
  generating_scene: "bg-emerald-400 animate-pulse",
  compositing: "bg-emerald-500 animate-pulse",
  exporting: "bg-indigo-400 animate-pulse",
  completed: "bg-green-500",
  failed: "bg-red-500",
};

export function isProcessing(status: JobStatus): boolean {
  return !["pending", "completed", "failed"].includes(status);
}
