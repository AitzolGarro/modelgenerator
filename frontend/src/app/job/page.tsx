"use client";

import { Suspense, useEffect, useState, useCallback } from "react";
import { useSearchParams, useRouter } from "next/navigation";
import dynamic from "next/dynamic";
import { getJob, deleteJob, retryJob, createJob } from "@/lib/api";
import type { Job } from "@/types/job";
import { isProcessing, JOB_TYPE_LABELS } from "@/types/job";
import StatusBadge from "@/components/StatusBadge";
import SpritePreview from "@/components/SpritePreview";

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
      const newJob = await createJob({
        job_type: type,
        prompt: actionPrompt.trim() || (type === "refine" ? "improve detail and quality" : "idle"),
        source_job_id: jobId,
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
          <SpritePreview
            spriteUrl={job.sprite_sheet_url}
            frameCount={24}
            frameWidth={512}
            frameHeight={512}
            fps={12}
            displaySize={320}
          />
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
        <div className="bg-violet-900/20 border border-violet-800 rounded-lg p-4 space-y-3">
          <h3 className="text-sm font-semibold text-violet-300">Animar este personaje 2D</h3>
          <div className="flex gap-2">
            <input
              type="text"
              value={actionPrompt}
              onChange={(e) => setActionPrompt(e.target.value)}
              placeholder="Ej: idle, walk cycle, attack, dance, wave, run..."
              className="flex-1 bg-gray-900 border border-violet-700 rounded px-3 py-2 text-sm text-white placeholder-gray-500 focus:outline-none focus:ring-1 focus:ring-violet-500"
            />
            <button
              onClick={() => handleAction("animate_2d")}
              disabled={actionLoading || !actionPrompt.trim()}
              className="px-4 py-2 text-sm bg-violet-600 hover:bg-violet-700 disabled:bg-gray-700 rounded whitespace-nowrap"
            >
              Generar Sprite
            </button>
          </div>
          <p className="text-xs text-violet-500">
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
