"use client";

import { Suspense, useEffect, useState, useCallback } from "react";
import { useSearchParams, useRouter } from "next/navigation";
import dynamic from "next/dynamic";
import { getJob, deleteJob, retryJob, createJob, getHealth } from "@/lib/api";
import type { Job } from "@/types/job";
import { isProcessing, JOB_TYPE_LABELS } from "@/types/job";
import StatusBadge from "@/components/StatusBadge";
import SpritePreview from "@/components/SpritePreview";

// ─── VRAM estimation (for animate_2d panel) ───────────────────────────────────
const VRAM_BASE_MB = 15360;
const FRAME_OVERHEAD_MB: Record<string, number> = { "480p": 6.4, "720p": 14.7 };
const SECONDS_PER_STEP_FRAME = 0.15;

function estimateVramMb(numFrames: number, resolution: string): number {
  return VRAM_BASE_MB + numFrames * (FRAME_OVERHEAD_MB[resolution] ?? 6.4);
}

const LEVEL_COLORS = {
  green:  { dot: "bg-green-500",  bar: "bg-green-500",  text: "text-green-400" },
  yellow: { dot: "bg-yellow-400", bar: "bg-yellow-400", text: "text-yellow-400" },
  red:    { dot: "bg-red-500",    bar: "bg-red-500",    text: "text-red-400" },
};

function vramLevelColor(estimatedMb: number, totalMb: number) {
  const ratio = estimatedMb / totalMb;
  if (ratio < 0.8) return LEVEL_COLORS.green;
  if (ratio < 0.95) return LEVEL_COLORS.yellow;
  return LEVEL_COLORS.red;
}

const ANIM_PRESETS_JOB = [
  { key: "rapido",     label: "Rápido",      desc: "~1 min",   detail: "17 frames · 20 pasos · 480p",  frames: 17, steps: 20, guidance: 7.0, res: "480p" },
  { key: "calidad",   label: "Calidad",      desc: "~2.5 min", detail: "33 frames · 30 pasos · 480p",  frames: 33, steps: 30, guidance: 9.0, res: "480p" },
  { key: "alta",      label: "Alta calidad", desc: "~5 min",   detail: "49 frames · 40 pasos · 480p",  frames: 49, steps: 40, guidance: 9.0, res: "480p" },
  { key: "cinematico",label: "Cinemático",   desc: "~10 min",  detail: "49 frames · 40 pasos · 720p",  frames: 49, steps: 40, guidance: 9.0, res: "720p" },
] as const;

const ModelViewer = dynamic(() => import("@/components/ModelViewer"), {
  ssr: false,
  loading: () => (
    <div className="w-full aspect-square bg-gray-900 rounded-lg flex items-center justify-center">
      <span className="text-gray-500">Cargando visor 3D...</span>
    </div>
  ),
});

export default function JobDetailPage() {
  return (
    <Suspense fallback={<div className="max-w-4xl mx-auto"><div className="h-96 bg-gray-900 rounded-lg animate-pulse" /></div>}>
      <JobDetailContent />
    </Suspense>
  );
}

function JobDetailContent() {
  const searchParams = useSearchParams();
  const router = useRouter();
  const jobId = Number(searchParams.get("id"));

  const [job, setJob] = useState<Job | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [activeTab, setActiveTab] = useState<"image" | "model">("image");
  const [actionPrompt, setActionPrompt] = useState("");
  const [actionLoading, setActionLoading] = useState(false);

  // animate_2d preset state
  const [animPresetKey, setAnimPresetKey] = useState<string>("calidad");
  const [gpuName, setGpuName] = useState<string | null>(null);
  const [gpuTotalMb, setGpuTotalMb] = useState<number | null>(null);

  useEffect(() => {
    getHealth()
      .then((h) => {
        setGpuName(h.gpu_name ?? null);
        setGpuTotalMb(h.gpu_memory_total_mb ?? null);
      })
      .catch(() => {});
  }, []);

  const loadJob = useCallback(async () => {
    if (!jobId) { setError("No job ID"); setLoading(false); return; }
    try {
      const data = await getJob(jobId);
      setJob(data);
      const is2d = data.job_type === "generate_2d" || data.job_type === "animate_2d";
      if (data.export_url && activeTab === "image" && !is2d) setActiveTab("model");
    } catch (err) {
      setError(err instanceof Error ? err.message : "Error loading job");
    } finally {
      setLoading(false);
    }
  }, [jobId, activeTab]);

  useEffect(() => {
    loadJob();
    const interval = setInterval(() => {
      if (job && (isProcessing(job.status) || job.status === "pending")) loadJob();
    }, 3000);
    return () => clearInterval(interval);
  }, [loadJob, job?.status]);

  async function handleAction(type: "animate" | "refine" | "animate_2d") {
    if (!actionPrompt.trim() && type === "animate") return;
    setActionLoading(true);
    try {
      const preset = ANIM_PRESETS_JOB.find((p) => p.key === animPresetKey) ?? ANIM_PRESETS_JOB[1];
      const newJob = await createJob({
        job_type: type,
        prompt: actionPrompt.trim() || (type === "refine" ? "improve detail and quality" : "idle"),
        source_job_id: jobId,
        ...(type === "animate_2d" ? {
          num_frames: preset.frames,
          anim_inference_steps: preset.steps,
          anim_guidance_scale: preset.guidance,
          anim_resolution: preset.res,
        } : {}),
      });
      router.push(`/job?id=${newJob.id}`);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Error creating job");
    } finally {
      setActionLoading(false);
    }
  }

  if (loading) return <div className="max-w-4xl mx-auto"><div className="h-96 bg-gray-900 rounded-lg animate-pulse" /></div>;

  if (error || !job) {
    return (
      <div className="max-w-4xl mx-auto text-center py-12">
        <p className="text-red-400">{error || "Job no encontrado"}</p>
        <button onClick={() => router.push("/")} className="mt-4 text-sm text-brand-400 hover:underline">Volver</button>
      </div>
    );
  }

  return (
    <div className="max-w-4xl mx-auto space-y-6">
      {/* Header */}
      <div className="flex items-start justify-between">
        <div>
          <div className="flex items-center gap-3 mb-2">
            <span className="text-xs uppercase tracking-wider text-gray-500 bg-gray-800 px-2 py-1 rounded">
              {JOB_TYPE_LABELS[job.job_type] || job.job_type}
            </span>
            <h1 className="text-xl font-bold">Job #{job.id}</h1>
            <StatusBadge status={job.status} />
          </div>
          <p className="text-gray-300">{job.prompt}</p>
        </div>
        <div className="flex gap-2">
          {job.status === "failed" && (
            <button onClick={() => retryJob(jobId).then(setJob)} className="px-3 py-1.5 text-sm bg-yellow-600 hover:bg-yellow-700 rounded">Reintentar</button>
          )}
          {(job.status === "completed" || job.status === "failed") && (
            <button onClick={async () => { await deleteJob(jobId); router.push("/jobs"); }} className="px-3 py-1.5 text-sm bg-red-900 hover:bg-red-800 border border-red-700 rounded">Eliminar</button>
          )}
        </div>
      </div>

      {/* Error */}
      {job.error_message && (
        <div className="bg-red-900/20 border border-red-800 rounded-lg p-4">
          <p className="text-sm text-red-400 font-mono">{job.error_message}</p>
        </div>
      )}

      {/* Preview */}
      {(job.image_url || job.export_url || job.sprite_sheet_url) && (
        <div>
          <div className="flex border-b border-gray-800 mb-4">
            {job.image_url && (
              <button onClick={() => setActiveTab("image")} className={`px-4 py-2 text-sm border-b-2 transition-colors ${activeTab === "image" ? "border-brand-500 text-white" : "border-transparent text-gray-500 hover:text-gray-300"}`}>
                {job.job_type === "scene" ? "Preview" : job.job_type === "generate_2d" ? "Personaje 2D" : "Imagen"}
              </button>
            )}
            {job.export_url && job.job_type !== "generate_2d" && job.job_type !== "animate_2d" && (
              <button onClick={() => setActiveTab("model")} className={`px-4 py-2 text-sm border-b-2 transition-colors ${activeTab === "model" ? "border-brand-500 text-white" : "border-transparent text-gray-500 hover:text-gray-300"}`}>
                {job.job_type === "animate" ? "Modelo animado" : job.job_type === "scene" ? "Escenario 3D" : "Modelo 3D"}
              </button>
            )}
          </div>

          {activeTab === "image" && job.image_url && (
            <div className="flex justify-center">
              <img src={job.image_url} alt={job.prompt} className="max-h-[500px] rounded-lg border border-gray-800" />
            </div>
          )}
          {activeTab === "model" && job.export_url && job.job_type !== "generate_2d" && job.job_type !== "animate_2d" && (
            <ModelViewer url={job.export_url} className="w-full aspect-square max-h-[600px]" />
          )}
        </div>
      )}

      {/* animate_2d: sprite sheet player */}
      {job.job_type === "animate_2d" && job.status === "completed" && job.sprite_sheet_url && (
        <div className="bg-gray-900/50 border border-gray-800 rounded-lg p-6 flex flex-col items-center gap-4">
          <h3 className="text-sm font-semibold text-gray-300 self-start">Vista previa de la animación</h3>
          {job.model_json_url ? (
            <SpritePreview
              spriteUrl={job.sprite_sheet_url}
              animationJsonUrl={job.model_json_url}
            />
          ) : (
            <div className="w-[320px] h-[320px] bg-gray-800 rounded-lg animate-pulse flex items-center justify-center">
              <span className="text-xs text-gray-500">Cargando metadata…</span>
            </div>
          )}
          <p className="text-xs text-gray-500">
            Sprite sheet listo para usar en motores de juego (Godot, Unity, etc.)
          </p>
        </div>
      )}

      {/* Processing */}
      {(isProcessing(job.status) || job.status === "pending") && (
        <div className="flex items-center justify-center py-12">
          <div className="text-center space-y-3">
            <div className="w-8 h-8 border-2 border-brand-500 border-t-transparent rounded-full animate-spin mx-auto" />
            <p className="text-sm text-gray-400">Procesando...</p>
          </div>
        </div>
      )}

      {/* Actions: Animate / Refine this 3D model */}
      {job.status === "completed" && job.export_url && job.job_type !== "generate_2d" && job.job_type !== "animate_2d" && (
        <div className="bg-gray-900/50 border border-gray-800 rounded-lg p-4 space-y-3">
          <h3 className="text-sm font-semibold text-gray-300">Acciones sobre este modelo 3D</h3>
          <div className="flex gap-2">
            <input
              type="text"
              value={actionPrompt}
              onChange={(e) => setActionPrompt(e.target.value)}
              placeholder="Ej: walking cycle, idle breathing, dance..."
              className="flex-1 bg-gray-900 border border-gray-700 rounded px-3 py-2 text-sm text-white placeholder-gray-500 focus:outline-none focus:ring-1 focus:ring-brand-500"
            />
            <button
              onClick={() => handleAction("animate")}
              disabled={actionLoading || !actionPrompt.trim()}
              className="px-4 py-2 text-sm bg-orange-600 hover:bg-orange-700 disabled:bg-gray-700 rounded whitespace-nowrap"
            >
              Animar
            </button>
            <button
              onClick={() => handleAction("refine")}
              disabled={actionLoading}
              className="px-4 py-2 text-sm bg-teal-600 hover:bg-teal-700 disabled:bg-gray-700 rounded whitespace-nowrap"
            >
              Mejorar
            </button>
          </div>
        </div>
      )}

      {/* Actions: Animate this 2D character */}
      {job.status === "completed" && job.job_type === "generate_2d" && (
        <div className="bg-violet-900/20 border border-violet-800/60 rounded-xl p-5 space-y-4">
          <h3 className="text-sm font-semibold text-violet-300">Animar este personaje 2D</h3>

          {/* Quality preset cards */}
          <div>
            <p className="text-xs text-gray-500 mb-2">Calidad de animación</p>
            <div className="grid grid-cols-2 sm:grid-cols-4 gap-2">
              {ANIM_PRESETS_JOB.map((preset) => {
                const presetVram = estimateVramMb(preset.frames, preset.res);
                const isAvailable = gpuTotalMb == null || presetVram <= gpuTotalMb;
                const isSelected = animPresetKey === preset.key;
                return (
                  <button
                    key={preset.key}
                    type="button"
                    disabled={!isAvailable || actionLoading}
                    onClick={() => setAnimPresetKey(preset.key)}
                    title={!isAvailable ? `Necesita ~${(presetVram / 1024).toFixed(1)} GB VRAM` : ""}
                    className={`p-3 rounded-xl border text-left transition-all duration-200 ${
                      !isAvailable
                        ? "bg-gray-900/30 border-gray-800 opacity-40 cursor-not-allowed"
                        : isSelected
                        ? "bg-gradient-to-br from-violet-900/50 to-gray-900 border-violet-500 shadow-md shadow-violet-500/20"
                        : "bg-gray-900/40 border-violet-800/30 hover:border-violet-500/50 hover:bg-violet-900/20 hover:scale-[1.02]"
                    }`}
                  >
                    <div className="text-sm font-semibold text-white">{preset.label}</div>
                    <div className={`text-xs mt-0.5 font-medium ${isSelected ? "text-violet-400" : "text-gray-400"}`}>
                      {preset.desc}
                    </div>
                    <div className="text-xs text-gray-600 mt-0.5">{preset.detail}</div>
                    {!isAvailable && (
                      <div className="text-xs mt-1 text-red-400">VRAM insuficiente</div>
                    )}
                  </button>
                );
              })}
            </div>
          </div>

          {/* VRAM widget */}
          {(() => {
            const preset = ANIM_PRESETS_JOB.find((p) => p.key === animPresetKey) ?? ANIM_PRESETS_JOB[1];
            const estimatedVramMb = estimateVramMb(preset.frames, preset.res);
            const estimatedTimeSec = preset.frames * preset.steps * SECONDS_PER_STEP_FRAME;
            const mins = Math.floor(estimatedTimeSec / 60);
            const secs = Math.round(estimatedTimeSec % 60);
            const timeLabel = mins === 0 ? `~${secs} seg` : `~${mins} min ${secs} seg`;
            const level = gpuTotalMb != null ? vramLevelColor(estimatedVramMb, gpuTotalMb) : null;
            const vramPct = gpuTotalMb != null ? Math.min(100, (estimatedVramMb / gpuTotalMb) * 100) : null;
            return (
              <div className="rounded-lg bg-black/30 border border-violet-900/30 px-3 py-2.5 space-y-2">
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-1.5">
                    <span className={`inline-block w-2 h-2 rounded-full ${level ? level.dot : "bg-gray-600"}`} />
                    <span className="text-xs text-gray-400">VRAM estimada</span>
                  </div>
                  <span className="text-xs text-gray-500">{timeLabel}</span>
                </div>
                <div className="flex items-center justify-between text-xs">
                  <span className="text-gray-600 truncate max-w-[55%]">{gpuName ?? "GPU desconocida"}</span>
                  <span className={level ? level.text : "text-gray-400"}>
                    {(estimatedVramMb / 1024).toFixed(1)} GB
                    {gpuTotalMb != null && <span className="text-gray-600"> / {(gpuTotalMb / 1024).toFixed(1)} GB</span>}
                  </span>
                </div>
                {vramPct != null && level && (
                  <div className="w-full h-1 rounded-full bg-white/5 overflow-hidden">
                    <div className={`h-full rounded-full transition-all ${level.bar}`} style={{ width: `${vramPct}%` }} />
                  </div>
                )}
              </div>
            );
          })()}

          {/* Animation prompt input */}
          <div className="flex gap-2">
            <input
              type="text"
              value={actionPrompt}
              onChange={(e) => setActionPrompt(e.target.value)}
              placeholder="Ej: idle, walk cycle, attack, dance, wave, run..."
              className="flex-1 bg-gray-900/60 border border-violet-700/50 rounded-lg px-3 py-2 text-sm text-white placeholder-gray-500 focus:outline-none focus:ring-1 focus:ring-violet-500"
            />
            <button
              onClick={() => handleAction("animate_2d")}
              disabled={actionLoading || !actionPrompt.trim()}
              className="px-4 py-2 text-sm bg-violet-600 hover:bg-violet-700 disabled:bg-gray-700 disabled:cursor-not-allowed rounded-lg whitespace-nowrap font-medium transition-colors"
            >
              {actionLoading ? "..." : "Generar Sprite"}
            </button>
          </div>
          <p className="text-xs text-violet-600">
            Tipos disponibles: idle, walk, run, attack, jump, dance, wave, hurt
          </p>
        </div>
      )}

      {/* Meta + Downloads */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <div className="bg-gray-900/50 border border-gray-800 rounded-lg p-4 space-y-2">
          <h3 className="text-sm font-semibold text-gray-300 mb-3">Detalles</h3>
          <Detail label="Tipo" value={JOB_TYPE_LABELS[job.job_type] || job.job_type} />
          {job.style && <Detail label="Estilo" value={job.style} />}
          <Detail label="Steps" value={job.num_steps} />
          <Detail label="Guidance" value={job.guidance_scale} />
          <Detail label="Seed" value={job.seed ?? "Aleatorio"} />
          <Detail label="Creado" value={new Date(job.created_at).toLocaleString()} />
          {job.completed_at && <Detail label="Completado" value={new Date(job.completed_at).toLocaleString()} />}
        </div>
        <div className="bg-gray-900/50 border border-gray-800 rounded-lg p-4">
          <h3 className="text-sm font-semibold text-gray-300 mb-3">Descargas</h3>
          <div className="space-y-2">
            {job.image_url && <DL href={job.image_url} label={job.job_type === "generate_2d" ? "Personaje 2D" : job.job_type === "animate_2d" ? "Sprite Sheet" : "Imagen"} ext="PNG" />}
            {job.export_url && job.job_type !== "generate_2d" && job.job_type !== "animate_2d" && (
              <DL href={job.export_url} label={job.job_type === "animate" ? "GLB animado" : job.job_type === "scene" ? "Escenario" : "Modelo 3D"} ext="GLB" />
            )}
            {job.sprite_sheet_url && <DL href={job.sprite_sheet_url} label="Sprite Sheet" ext="PNG" />}
            {job.model_json_url && job.job_type === "animate_2d" && <DL href={job.model_json_url} label="Metadata animación" ext="JSON" />}
            {job.model_url && job.model_url !== job.export_url && job.job_type !== "animate_2d" && (
              <DL href={job.model_url} label={job.job_type === "generate_2d" ? "Model JSON" : "Mesh original"} ext={job.job_type === "generate_2d" ? "JSON" : "OBJ"} />
            )}
          </div>
          {!job.image_url && !job.export_url && !job.sprite_sheet_url && <p className="text-sm text-gray-500">Disponible al completar.</p>}
        </div>
      </div>
    </div>
  );
}

function Detail({ label, value }: { label: string; value: string | number }) {
  return (<div className="flex justify-between text-sm"><span className="text-gray-500">{label}</span><span className="text-gray-300">{value}</span></div>);
}

function DL({ href, label, ext }: { href: string; label: string; ext: string }) {
  return (
    <a href={href} download className="flex items-center justify-between p-2 bg-gray-800 hover:bg-gray-700 rounded transition-colors">
      <span className="text-sm">{label}</span>
      <span className="text-xs text-gray-400 bg-gray-900 px-2 py-0.5 rounded">{ext}</span>
    </a>
  );
}
