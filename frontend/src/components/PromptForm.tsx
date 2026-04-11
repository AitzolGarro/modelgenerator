"use client";

import { useState, useRef, useEffect } from "react";
import { useRouter } from "next/navigation";
import { createJob, uploadAndCreateJob, getHealth } from "@/lib/api";
import type { JobType, JobCreatePayload } from "@/types/job";
import { STYLE_OPTIONS } from "@/types/job";

const JOB_TYPES: { value: JobType; label: string; description: string; needsFile: boolean; is2D?: boolean }[] = [
  { value: "generate",    label: "Generar 3D",   description: "Texto → Imagen → Modelo 3D",       needsFile: false },
  { value: "animate",     label: "Animar 3D",    description: "GLB + prompt → GLB animado",        needsFile: true },
  { value: "refine",      label: "Mejorar",      description: "GLB → mas detalle y calidad",       needsFile: true },
  { value: "scene",       label: "Escenario",    description: "Texto → Entorno 3D completo",       needsFile: false },
  { value: "skin",        label: "Texturizar",   description: "GLB + prompt → GLB texturizado",    needsFile: true },
  { value: "generate_2d", label: "Generar 2D",   description: "Texto → Personaje 2D articulado",  needsFile: false, is2D: true },
  { value: "animate_2d",  label: "Animar 2D",    description: "ID de job 2D + prompt → sprite",   needsFile: false, is2D: true },
];

// ─── VRAM estimation constants ────────────────────────────────────────────────
const VRAM_BASE_MB = 15360; // 15 GB for Wan2.1 14B int4 with cpu_offload
const FRAME_OVERHEAD_MB: Record<string, number> = { "480p": 6.4, "720p": 14.7 };
const SECONDS_PER_STEP_FRAME = 0.15; // empirical factor for RTX 5090 int4

function estimateVramMb(numFrames: number, resolution: string): number {
  const frameOverhead = FRAME_OVERHEAD_MB[resolution] ?? 6.4;
  return VRAM_BASE_MB + numFrames * frameOverhead;
}

function estimateSeconds(numFrames: number, steps: number): number {
  return numFrames * steps * SECONDS_PER_STEP_FRAME;
}

function formatDuration(totalSeconds: number): string {
  const mins = Math.floor(totalSeconds / 60);
  const secs = Math.round(totalSeconds % 60);
  if (mins === 0) return `~${secs} sec`;
  return `~${mins} min ${secs} sec`;
}

function vramLevel(estimatedMb: number, totalMb: number): "green" | "yellow" | "red" {
  const ratio = estimatedMb / totalMb;
  if (ratio < 0.8) return "green";
  if (ratio < 0.95) return "yellow";
  return "red";
}

// ─── Animation param defaults ─────────────────────────────────────────────────
interface AnimationParams {
  num_frames: number;
  anim_guidance_scale: number;
  anim_inference_steps: number;
  anim_resolution: string;
  enhance_animation: boolean;
  enhance_personality: string;
  enhance_intensity: number;
}

const ANIM_DEFAULTS: AnimationParams = {
  num_frames: 33,
  anim_guidance_scale: 9.0,
  anim_inference_steps: 30,
  anim_resolution: "480p",
  enhance_animation: false,
  enhance_personality: "calm",
  enhance_intensity: 0.7,
};

// Valid 4k+1 frame counts in the range 17–81
const VALID_FRAME_COUNTS = [17, 21, 25, 29, 33, 37, 41, 45, 49, 53, 57, 61, 65, 69, 73, 77, 81];

const PERSONALITY_OPTIONS = [
  { value: "calm",       label: "Calm" },
  { value: "aggressive", label: "Agresivo" },
  { value: "heavy",      label: "Pesado" },
  { value: "light",      label: "Ligero" },
];

// ─── Traffic-light colors ─────────────────────────────────────────────────────
const LEVEL_COLORS = {
  green:  { dot: "bg-green-500",  bar: "bg-green-500",  text: "text-green-400" },
  yellow: { dot: "bg-yellow-400", bar: "bg-yellow-400", text: "text-yellow-400" },
  red:    { dot: "bg-red-500",    bar: "bg-red-500",    text: "text-red-400" },
};

// ─── Component ────────────────────────────────────────────────────────────────
export default function PromptForm() {
  const router = useRouter();
  const fileRef = useRef<HTMLInputElement>(null);

  const [jobType, setJobType] = useState<JobType>("generate");
  const [prompt, setPrompt] = useState("");
  const [negativePrompt, setNegativePrompt] = useState("");
  const [numSteps, setNumSteps] = useState(30);
  const [guidanceScale, setGuidanceScale] = useState(7.5);
  const [seed, setSeed] = useState("");
  const [file, setFile] = useState<File | null>(null);
  const [style2D, setStyle2D] = useState("anime");
  const [sourceJobId, setSourceJobId] = useState("");
  const [showAdvanced, setShowAdvanced] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Task 2.2 — Animation params state
  const [animationParams, setAnimationParams] = useState<AnimationParams>(ANIM_DEFAULTS);
  const [showAnimOptions, setShowAnimOptions] = useState(true);

  // Task 3.1 — GPU info state
  const [gpuName, setGpuName] = useState<string | null>(null);
  const [gpuTotalMb, setGpuTotalMb] = useState<number | null>(null);

  // Fetch health on mount (Task 3.1)
  useEffect(() => {
    getHealth()
      .then((h) => {
        setGpuName(h.gpu_name ?? null);
        setGpuTotalMb(h.gpu_memory_total_mb ?? null);
      })
      .catch(() => {
        // Graceful degradation — no color coding
      });
  }, []);

  const selectedType = JOB_TYPES.find(t => t.value === jobType)!;

  function updateAnim<K extends keyof AnimationParams>(key: K, value: AnimationParams[K]) {
    setAnimationParams((prev) => ({ ...prev, [key]: value }));
  }

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    if (!prompt.trim()) return;

    setLoading(true);
    setError(null);

    try {
      let job;

      if (selectedType.needsFile && file) {
        // Upload mode (animate/refine/skin)
        job = await uploadAndCreateJob(
          file,
          jobType as "animate" | "refine" | "skin",
          prompt.trim(),
          negativePrompt.trim() || undefined,
        );
      } else {
        // JSON mode
        const payload: JobCreatePayload = {
          job_type: jobType,
          prompt: prompt.trim(),
          num_steps: numSteps,
          guidance_scale: guidanceScale,
        };
        if (negativePrompt.trim()) payload.negative_prompt = negativePrompt.trim();
        if (seed.trim()) payload.seed = parseInt(seed, 10);

        // 2D-specific
        if (jobType === "generate_2d") {
          payload.style = style2D;
        }
        if (jobType === "animate_2d") {
          const sid = parseInt(sourceJobId, 10);
          if (!sid || isNaN(sid)) {
            setError("Ingresa un ID de job generate_2d válido");
            setLoading(false);
            return;
          }
          payload.source_job_id = sid;

          // Task 2.8 — Include animation params in payload
          payload.num_frames = animationParams.num_frames;
          payload.anim_inference_steps = animationParams.anim_inference_steps;
          payload.anim_guidance_scale = animationParams.anim_guidance_scale;
          payload.anim_resolution = animationParams.anim_resolution;
          payload.enhance_animation = animationParams.enhance_animation;
          payload.enhance_personality = animationParams.enhance_personality;
          payload.enhance_intensity = animationParams.enhance_intensity;
        }

        job = await createJob(payload);
      }

      router.push(`/job?id=${job.id}`);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Error creating job");
    } finally {
      setLoading(false);
    }
  }

  const promptPlaceholders: Record<JobType, string> = {
    generate:    "Ej: A detailed medieval sword with ornate handle and gemstone",
    animate:     "Ej: walking cycle, arms swinging naturally",
    refine:      "Ej: increase detail, smooth surface, fix normals",
    scene:       "Ej: enchanted forest with glowing mushrooms and a small lake",
    skin:        "Ej: weathered stone texture with mossy cracks, dark fantasy style",
    generate_2d: "Ej: A female warrior with blue armor, silver hair, determined expression",
    animate_2d:  "Ej: idle breathing, walk cycle, attack swing",
  };

  // ─── VRAM computations (Tasks 3.2, 3.3, 3.4) ─────────────────────────────
  const estimatedVramMb = estimateVramMb(animationParams.num_frames, animationParams.anim_resolution);
  const estimatedTimeSec = estimateSeconds(animationParams.num_frames, animationParams.anim_inference_steps);
  const timeLabel = formatDuration(estimatedTimeSec);

  const level = gpuTotalMb != null ? vramLevel(estimatedVramMb, gpuTotalMb) : null;
  const levelColors = level ? LEVEL_COLORS[level] : null;

  const vramPct = gpuTotalMb != null ? Math.min(100, (estimatedVramMb / gpuTotalMb) * 100) : null;

  return (
    <form onSubmit={handleSubmit} className="space-y-4">
      {/* Job type selector */}
      <div className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-7 gap-2">
        {JOB_TYPES.map((t) => (
          <button
            key={t.value}
            type="button"
            onClick={() => { setJobType(t.value); setFile(null); }}
            className={`p-3 rounded-lg border text-left transition-all ${
              jobType === t.value
                ? t.is2D
                  ? "bg-violet-600/20 border-violet-500 text-white"
                  : "bg-brand-600/20 border-brand-500 text-white"
                : "bg-gray-900 border-gray-700 text-gray-400 hover:border-gray-500"
            }`}
          >
            <div className="text-sm font-medium">{t.label}</div>
            <div className="text-xs mt-0.5 opacity-70">{t.description}</div>
          </button>
        ))}
      </div>

      {/* File upload for animate/refine/skin */}
      {selectedType.needsFile && (
        <div>
          <label className="block text-sm text-gray-300 mb-1">Archivo GLB de entrada</label>
          <div
            onClick={() => fileRef.current?.click()}
            className={`border-2 border-dashed rounded-lg p-4 text-center cursor-pointer transition-colors ${
              file ? "border-brand-500 bg-brand-600/10" : "border-gray-700 hover:border-gray-500"
            }`}
          >
            <input
              ref={fileRef}
              type="file"
              accept=".glb"
              onChange={(e) => setFile(e.target.files?.[0] || null)}
              className="hidden"
            />
            {file ? (
              <div>
                <p className="text-sm text-white">{file.name}</p>
                <p className="text-xs text-gray-400">{(file.size / 1024 / 1024).toFixed(1)} MB</p>
              </div>
            ) : (
              <p className="text-sm text-gray-500">Click para seleccionar .glb</p>
            )}
          </div>
        </div>
      )}

      {/* 2D style selector — only for generate_2d */}
      {jobType === "generate_2d" && (
        <div>
          <label className="block text-sm text-gray-300 mb-1">Estilo artístico</label>
          <select
            value={style2D}
            onChange={(e) => setStyle2D(e.target.value)}
            className="w-full bg-gray-900 border border-gray-700 rounded-lg px-3 py-2 text-white focus:outline-none focus:ring-2 focus:ring-violet-500"
            disabled={loading}
          >
            {STYLE_OPTIONS.map((opt) => (
              <option key={opt.value} value={opt.value}>{opt.label}</option>
            ))}
          </select>
        </div>
      )}

      {/* Source job ID — only for animate_2d */}
      {jobType === "animate_2d" && (
        <div>
          <label className="block text-sm text-gray-300 mb-1">
            ID del job <span className="text-violet-400 font-medium">Generar 2D</span> de origen
          </label>
          <input
            type="number"
            value={sourceJobId}
            onChange={(e) => setSourceJobId(e.target.value)}
            placeholder="Ej: 42"
            min={1}
            className="w-full bg-gray-900 border border-gray-700 rounded-lg px-4 py-2 text-white placeholder-gray-500 focus:outline-none focus:ring-2 focus:ring-violet-500"
            disabled={loading}
          />
          <p className="text-xs text-gray-500 mt-1">
            Puedes encontrar el ID en la URL del job completado: /job?id=42
          </p>
        </div>
      )}

      {/* Task 2.3 — Collapsible "Opciones de animación" — only for animate_2d */}
      {jobType === "animate_2d" && (
        <div className="rounded-lg border border-violet-800/60 bg-gray-900/60 overflow-hidden">
          {/* Panel header */}
          <button
            type="button"
            onClick={() => setShowAnimOptions(!showAnimOptions)}
            className="w-full flex items-center justify-between px-4 py-3 text-sm font-medium text-violet-300 hover:text-white hover:bg-violet-900/20 transition-colors"
          >
            <span>Opciones de animación</span>
            <span className="text-lg leading-none">{showAnimOptions ? "▲" : "▼"}</span>
          </button>

          {showAnimOptions && (
            <div className="px-4 pb-4 space-y-4 border-t border-violet-800/40">

              {/* Task 2.4 — num_frames dropdown */}
              <div className="grid grid-cols-2 gap-4 pt-3">
                <div>
                  <label className="block text-xs text-gray-400 mb-1">
                    Fotogramas
                  </label>
                  <select
                    value={animationParams.num_frames}
                    onChange={(e) => updateAnim("num_frames", parseInt(e.target.value))}
                    disabled={loading}
                    className="w-full bg-gray-800 border border-gray-700 rounded px-3 py-2 text-sm text-white focus:outline-none focus:ring-1 focus:ring-violet-500 disabled:opacity-50"
                  >
                    {VALID_FRAME_COUNTS.map((n) => (
                      <option key={n} value={n}>{n}</option>
                    ))}
                  </select>
                </div>

                {/* Task 2.7 — resolution toggle */}
                <div>
                  <label className="block text-xs text-gray-400 mb-1">Resolución</label>
                  <div className="flex rounded overflow-hidden border border-gray-700">
                    {["480p", "720p"].map((res) => (
                      <button
                        key={res}
                        type="button"
                        onClick={() => updateAnim("anim_resolution", res)}
                        disabled={loading}
                        className={`flex-1 py-2 text-sm font-medium transition-colors ${
                          animationParams.anim_resolution === res
                            ? "bg-violet-600 text-white"
                            : "bg-gray-800 text-gray-400 hover:bg-gray-700"
                        } disabled:opacity-50`}
                      >
                        {res}
                      </button>
                    ))}
                  </div>
                </div>
              </div>

              {/* Task 2.5 — guidance_scale slider */}
              <div>
                <label className="block text-xs text-gray-400 mb-1">
                  Escala de guía: <span className="text-white font-medium">{animationParams.anim_guidance_scale.toFixed(1)}</span>
                </label>
                <input
                  type="range"
                  min={1.0}
                  max={15.0}
                  step={0.5}
                  value={animationParams.anim_guidance_scale}
                  onChange={(e) => updateAnim("anim_guidance_scale", parseFloat(e.target.value))}
                  disabled={loading}
                  className="w-full accent-violet-500 disabled:opacity-50"
                />
                <div className="flex justify-between text-xs text-gray-600 mt-0.5">
                  <span>1.0</span><span>15.0</span>
                </div>
              </div>

              {/* Task 2.6 — inference_steps slider */}
              <div>
                <label className="block text-xs text-gray-400 mb-1">
                  Pasos de inferencia: <span className="text-white font-medium">{animationParams.anim_inference_steps}</span>
                </label>
                <input
                  type="range"
                  min={10}
                  max={50}
                  step={1}
                  value={animationParams.anim_inference_steps}
                  onChange={(e) => updateAnim("anim_inference_steps", parseInt(e.target.value))}
                  disabled={loading}
                  className="w-full accent-violet-500 disabled:opacity-50"
                />
                <div className="flex justify-between text-xs text-gray-600 mt-0.5">
                  <span>10</span><span>50</span>
                </div>
              </div>

              {/* Task 2.7 — enhance controls */}
              <div className="space-y-3 pt-1 border-t border-gray-800">
                <div className="flex items-center gap-2 pt-2">
                  <input
                    type="checkbox"
                    id="enhance_animation"
                    checked={animationParams.enhance_animation}
                    onChange={(e) => updateAnim("enhance_animation", e.target.checked)}
                    disabled={loading}
                    className="w-4 h-4 accent-violet-500 disabled:opacity-50"
                  />
                  <label htmlFor="enhance_animation" className="text-sm text-gray-300 cursor-pointer">
                    Mejoras de animación
                  </label>
                </div>

                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <label className="block text-xs text-gray-400 mb-1">Personalidad</label>
                    <select
                      value={animationParams.enhance_personality}
                      onChange={(e) => updateAnim("enhance_personality", e.target.value)}
                      disabled={loading || !animationParams.enhance_animation}
                      className="w-full bg-gray-800 border border-gray-700 rounded px-3 py-2 text-sm text-white focus:outline-none focus:ring-1 focus:ring-violet-500 disabled:opacity-40 disabled:cursor-not-allowed"
                    >
                      {PERSONALITY_OPTIONS.map((opt) => (
                        <option key={opt.value} value={opt.value}>{opt.label}</option>
                      ))}
                    </select>
                  </div>

                  <div>
                    <label className="block text-xs text-gray-400 mb-1">
                      Intensidad: <span className={`font-medium ${animationParams.enhance_animation ? "text-white" : "text-gray-600"}`}>
                        {animationParams.enhance_intensity.toFixed(1)}
                      </span>
                    </label>
                    <input
                      type="range"
                      min={0.0}
                      max={1.0}
                      step={0.1}
                      value={animationParams.enhance_intensity}
                      onChange={(e) => updateAnim("enhance_intensity", parseFloat(e.target.value))}
                      disabled={loading || !animationParams.enhance_animation}
                      className="w-full accent-violet-500 disabled:opacity-40"
                    />
                    <div className="flex justify-between text-xs text-gray-600 mt-0.5">
                      <span>0.0</span><span>1.0</span>
                    </div>
                  </div>
                </div>
              </div>

              {/* Tasks 3.2–3.5 — VRAM Estimate Widget */}
              <div className="rounded-lg bg-gray-950/80 border border-gray-800 px-3 py-2.5 mt-2 space-y-2">
                {/* Header row */}
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-1.5">
                    {levelColors ? (
                      <span className={`inline-block w-2.5 h-2.5 rounded-full ${levelColors.dot}`} />
                    ) : (
                      <span className="inline-block w-2.5 h-2.5 rounded-full bg-gray-600" />
                    )}
                    <span className="text-xs font-medium text-gray-300">VRAM estimada</span>
                  </div>
                  <span className="text-xs text-gray-500">{timeLabel}</span>
                </div>

                {/* GPU name + memory numbers */}
                <div className="flex items-center justify-between text-xs">
                  <span className="text-gray-500 truncate max-w-[55%]">
                    {gpuName ?? "GPU desconocida"}
                  </span>
                  <span className={levelColors ? levelColors.text : "text-gray-400"}>
                    {(estimatedVramMb / 1024).toFixed(1)} GB
                    {gpuTotalMb != null && (
                      <span className="text-gray-600"> / {(gpuTotalMb / 1024).toFixed(1)} GB</span>
                    )}
                  </span>
                </div>

                {/* Progress bar — only when GPU total is known */}
                {vramPct != null && levelColors && (
                  <div className="w-full h-1.5 rounded-full bg-gray-800 overflow-hidden">
                    <div
                      className={`h-full rounded-full transition-all ${levelColors.bar}`}
                      style={{ width: `${vramPct}%` }}
                    />
                  </div>
                )}
              </div>

            </div>
          )}
        </div>
      )}

      {/* Prompt */}
      <div>
        <label htmlFor="prompt" className="block text-sm font-medium text-gray-300 mb-1">
          {jobType === "animate"     ? "Describe la animación" :
           jobType === "animate_2d"  ? "Describe la animación (idle, walk, attack, dance…)" :
           jobType === "refine"      ? "Instrucciones de mejora" :
           jobType === "scene"       ? "Describe el escenario" :
           jobType === "skin"        ? "Describe la textura / materiales" :
           jobType === "generate_2d" ? "Describe tu personaje 2D" :
           "Describe tu modelo 3D"}
        </label>
        <textarea
          id="prompt"
          value={prompt}
          onChange={(e) => setPrompt(e.target.value)}
          placeholder={promptPlaceholders[jobType]}
          rows={3}
          className="w-full bg-gray-900 border border-gray-700 rounded-lg px-4 py-3 text-white placeholder-gray-500 focus:outline-none focus:ring-2 focus:ring-brand-500 focus:border-transparent resize-none"
          disabled={loading}
          maxLength={2000}
        />
      </div>

      {/* Advanced options */}
      <button
        type="button"
        onClick={() => setShowAdvanced(!showAdvanced)}
        className="text-sm text-gray-400 hover:text-white transition-colors"
      >
        {showAdvanced ? "- Ocultar opciones" : "+ Opciones avanzadas"}
      </button>

      {showAdvanced && (
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4 p-4 bg-gray-900/50 rounded-lg border border-gray-800">
          <div>
            <label className="block text-sm text-gray-400 mb-1">Prompt negativo</label>
            <input
              type="text"
              value={negativePrompt}
              onChange={(e) => setNegativePrompt(e.target.value)}
              placeholder="Cosas que evitar..."
              className="w-full bg-gray-900 border border-gray-700 rounded px-3 py-2 text-sm text-white placeholder-gray-500 focus:outline-none focus:ring-1 focus:ring-brand-500"
              disabled={loading}
            />
          </div>
          <div>
            <label className="block text-sm text-gray-400 mb-1">Seed</label>
            <input
              type="number"
              value={seed}
              onChange={(e) => setSeed(e.target.value)}
              placeholder="Aleatorio"
              min={0}
              className="w-full bg-gray-900 border border-gray-700 rounded px-3 py-2 text-sm text-white placeholder-gray-500 focus:outline-none focus:ring-1 focus:ring-brand-500"
              disabled={loading}
            />
          </div>
          {(jobType === "generate" || jobType === "scene") && (
            <>
              <div>
                <label className="block text-sm text-gray-400 mb-1">Steps: {numSteps}</label>
                <input type="range" value={numSteps} onChange={(e) => setNumSteps(parseInt(e.target.value))} min={1} max={100} className="w-full accent-brand-500" disabled={loading} />
              </div>
              <div>
                <label className="block text-sm text-gray-400 mb-1">Guidance: {guidanceScale}</label>
                <input type="range" value={guidanceScale} onChange={(e) => setGuidanceScale(parseFloat(e.target.value))} min={1} max={20} step={0.5} className="w-full accent-brand-500" disabled={loading} />
              </div>
            </>
          )}
        </div>
      )}

      {error && (
        <div className="text-red-400 text-sm bg-red-900/20 border border-red-800 rounded px-3 py-2">{error}</div>
      )}

      <button
        type="submit"
        disabled={
          loading ||
          !prompt.trim() ||
          (selectedType.needsFile && !file) ||
          (jobType === "animate_2d" && !sourceJobId.trim())
        }
        className={`w-full ${
          selectedType.is2D
            ? "bg-violet-600 hover:bg-violet-700"
            : "bg-brand-600 hover:bg-brand-700"
        } disabled:bg-gray-700 disabled:cursor-not-allowed text-white font-medium py-3 px-6 rounded-lg transition-colors`}
      >
        {loading ? "Procesando..." : selectedType.label}
      </button>
    </form>
  );
}
