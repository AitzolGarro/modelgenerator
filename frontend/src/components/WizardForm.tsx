"use client";

import { useState, useRef, useEffect } from "react";
import { useRouter } from "next/navigation";
import { createJob, uploadAndCreateJob, getHealth } from "@/lib/api";
import type { JobType, JobCreatePayload } from "@/types/job";
import { STYLE_OPTIONS } from "@/types/job";
import {
  Box,
  Image,
  Mountain,
  Play,
  Sparkles,
  Palette,
  Film,
  ChevronDown,
  ChevronUp,
  type LucideIcon,
} from "lucide-react";

// ─── VRAM estimation constants ─────────────────────────────────────────────────
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

// ─── Animation param defaults ──────────────────────────────────────────────────
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

// ─── Traffic-light colors ──────────────────────────────────────────────────────
const LEVEL_COLORS = {
  green:  { dot: "bg-green-500",  bar: "bg-green-500",  text: "text-green-400" },
  yellow: { dot: "bg-yellow-400", bar: "bg-yellow-400", text: "text-yellow-400" },
  red:    { dot: "bg-red-500",    bar: "bg-red-500",    text: "text-red-400" },
};

// ─── Animation quality presets ─────────────────────────────────────────────────
const ANIM_PRESETS = [
  { key: "rapido",     label: "Rápido",       desc: "~1 min · 17 fotogramas",          frames: 17, steps: 20, guidance: 7.0, res: "480p" },
  { key: "calidad",   label: "Calidad",       desc: "~2.5 min · 33 fotogramas",        frames: 33, steps: 30, guidance: 9.0, res: "480p" },
  { key: "alta",      label: "Alta calidad",  desc: "~5 min · 49 fotogramas",          frames: 49, steps: 40, guidance: 9.0, res: "480p" },
  { key: "cinematico",label: "Cinemático",    desc: "~10 min · 49 fotogramas 720p",    frames: 49, steps: 40, guidance: 9.0, res: "720p" },
] as const;

// ─── Job type definitions ──────────────────────────────────────────────────────
interface JobTypeDef {
  value: JobType;
  label: string;
  description: string;
  needsFile: boolean;
  is2D?: boolean;
  icon: LucideIcon;
  group: "creation" | "postprocess";
}

const JOB_TYPE_DEFS: JobTypeDef[] = [
  { value: "generate",    label: "Generar 3D",    description: "Texto → Imagen → Modelo 3D",       needsFile: false, icon: Box,      group: "creation" },
  { value: "generate_2d", label: "Generar 2D",    description: "Texto → Personaje 2D articulado",  needsFile: false, is2D: true, icon: Image,    group: "creation" },
  { value: "scene",       label: "Escenario",     description: "Texto → Entorno 3D completo",       needsFile: false, icon: Mountain, group: "creation" },
  { value: "animate",     label: "Animar 3D",     description: "GLB + prompt → GLB animado",        needsFile: true,  icon: Play,     group: "postprocess" },
  { value: "refine",      label: "Mejorar",       description: "GLB → más detalle y calidad",       needsFile: true,  icon: Sparkles, group: "postprocess" },
  { value: "skin",        label: "Texturizar",    description: "GLB + prompt → GLB texturizado",    needsFile: true,  icon: Palette,  group: "postprocess" },
  { value: "animate_2d",  label: "Animar 2D",     description: "ID de job 2D + prompt → sprite",   needsFile: false, is2D: true, icon: Film,     group: "postprocess" },
];

// ─── Style definitions for generate_2d ────────────────────────────────────────
const STYLE_ICONS: Record<string, string> = {
  anime:     "🌸",
  pixel_art: "🕹️",
  cartoon:   "🎨",
  realistic: "📸",
  chibi:     "🌟",
  comic:     "💥",
};

// ─── Prompt placeholders ───────────────────────────────────────────────────────
const promptPlaceholders: Record<JobType, string> = {
  generate:    "Ej: A detailed medieval sword with ornate handle and gemstone",
  animate:     "Ej: walking cycle, arms swinging naturally",
  refine:      "Ej: increase detail, smooth surface, fix normals",
  scene:       "Ej: enchanted forest with glowing mushrooms and a small lake",
  skin:        "Ej: weathered stone texture with mossy cracks, dark fantasy style",
  generate_2d: "Ej: A female warrior with blue armor, silver hair, determined expression",
  animate_2d:  "Ej: idle breathing, walk cycle, attack swing",
};

// ─── Step indicator ────────────────────────────────────────────────────────────
function StepIndicator({ currentStep }: { currentStep: 1 | 2 | 3 }) {
  const steps = [
    { n: 1, label: "Tipo" },
    { n: 2, label: "Configurar" },
    { n: 3, label: "Revisar" },
  ];
  return (
    <div className="flex items-center justify-center gap-0 mb-8">
      {steps.map((s, i) => (
        <div key={s.n} className="flex items-center">
          <div className="flex flex-col items-center gap-1">
            <div
              className={`w-8 h-8 rounded-full flex items-center justify-center text-sm font-semibold transition-all duration-300 ${
                currentStep > s.n
                  ? "bg-violet-600 text-white"
                  : currentStep === s.n
                    ? "bg-violet-600 text-white ring-4 ring-violet-500/30"
                    : "bg-gray-800 text-gray-500"
              }`}
            >
              {currentStep > s.n ? (
                <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={3}>
                  <path strokeLinecap="round" strokeLinejoin="round" d="M5 13l4 4L19 7" />
                </svg>
              ) : (
                s.n
              )}
            </div>
            <span className={`text-xs ${currentStep === s.n ? "text-violet-300" : "text-gray-500"}`}>
              {s.label}
            </span>
          </div>
          {i < steps.length - 1 && (
            <div className={`w-16 h-px mx-1 mb-5 transition-colors duration-300 ${currentStep > s.n ? "bg-violet-600" : "bg-gray-800"}`} />
          )}
        </div>
      ))}
    </div>
  );
}

// ─── VRAM Widget (reusable) ────────────────────────────────────────────────────
function VramWidget({
  numFrames,
  steps,
  resolution,
  gpuName,
  gpuTotalMb,
}: {
  numFrames: number;
  steps: number;
  resolution: string;
  gpuName: string | null;
  gpuTotalMb: number | null;
}) {
  const estimatedVramMb = estimateVramMb(numFrames, resolution);
  const estimatedTimeSec = estimateSeconds(numFrames, steps);
  const timeLabel = formatDuration(estimatedTimeSec);
  const level = gpuTotalMb != null ? vramLevel(estimatedVramMb, gpuTotalMb) : null;
  const levelColors = level ? LEVEL_COLORS[level] : null;
  const vramPct = gpuTotalMb != null ? Math.min(100, (estimatedVramMb / gpuTotalMb) * 100) : null;

  return (
    <div className="rounded-lg bg-gray-950/80 border border-gray-800 px-3 py-2.5 space-y-2">
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
      <div className="flex items-center justify-between text-xs">
        <span className="text-gray-500 truncate max-w-[55%]">{gpuName ?? "GPU desconocida"}</span>
        <span className={levelColors ? levelColors.text : "text-gray-400"}>
          {(estimatedVramMb / 1024).toFixed(1)} GB
          {gpuTotalMb != null && (
            <span className="text-gray-600"> / {(gpuTotalMb / 1024).toFixed(1)} GB</span>
          )}
        </span>
      </div>
      {vramPct != null && levelColors && (
        <div className="w-full h-1.5 rounded-full bg-gray-800 overflow-hidden">
          <div
            className={`h-full rounded-full transition-all ${levelColors.bar}`}
            style={{ width: `${vramPct}%` }}
          />
        </div>
      )}
    </div>
  );
}

// ─── Main component ────────────────────────────────────────────────────────────
export default function WizardForm() {
  const router = useRouter();
  const fileRef = useRef<HTMLInputElement>(null);

  // ─ Step state
  const [step, setStep] = useState<1 | 2 | 3>(1);

  // ─ Form state
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
  const [showAnimOptions, setShowAnimOptions] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // ─ Animation params
  const [animationParams, setAnimationParams] = useState<AnimationParams>(ANIM_DEFAULTS);

  // ─ GPU info
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

  const selectedTypeDef = JOB_TYPE_DEFS.find((t) => t.value === jobType)!;

  function updateAnim<K extends keyof AnimationParams>(key: K, value: AnimationParams[K]) {
    setAnimationParams((prev) => ({ ...prev, [key]: value }));
  }

  function selectJobType(jt: JobType) {
    setJobType(jt);
    setFile(null);
    setError(null);
    setStep(2);
  }

  function goBack() {
    setError(null);
    setStep((s) => Math.max(1, s - 1) as 1 | 2 | 3);
  }

  function goNext() {
    setError(null);
    setStep((s) => Math.min(3, s + 1) as 1 | 2 | 3);
  }

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    if (!prompt.trim()) return;

    setLoading(true);
    setError(null);

    try {
      let job;

      if (selectedTypeDef.needsFile && file) {
        job = await uploadAndCreateJob(
          file,
          jobType as "animate" | "refine" | "skin",
          prompt.trim(),
          negativePrompt.trim() || undefined,
        );
      } else {
        const payload: JobCreatePayload = {
          job_type: jobType,
          prompt: prompt.trim(),
          num_steps: numSteps,
          guidance_scale: guidanceScale,
        };
        if (negativePrompt.trim()) payload.negative_prompt = negativePrompt.trim();
        if (seed.trim()) payload.seed = parseInt(seed, 10);

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

  const canProceedToStep3 =
    prompt.trim().length > 0 &&
    (!selectedTypeDef.needsFile || file !== null) &&
    (jobType !== "animate_2d" || sourceJobId.trim().length > 0);

  // ─── Step 1: Job Type Selection ─────────────────────────────────────────────
  const creationTypes = JOB_TYPE_DEFS.filter((t) => t.group === "creation");
  const postprocessTypes = JOB_TYPE_DEFS.filter((t) => t.group === "postprocess");

  const renderStep1 = () => (
    <div className="animate-slide-in space-y-6">
      {/* Creación group */}
      <div>
        <h3 className="text-xs font-semibold uppercase tracking-widest text-gray-500 mb-3">
          Creación
        </h3>
        <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
          {creationTypes.map((t) => {
            const Icon = t.icon;
            const isSelected = jobType === t.value;
            const accent = t.is2D ? "violet" : "brand";
            return (
              <button
                key={t.value}
                type="button"
                onClick={() => selectJobType(t.value)}
                className={`
                  p-4 rounded-xl border text-left transition-all duration-200
                  backdrop-blur-md
                  ${isSelected
                    ? t.is2D
                      ? "bg-gradient-to-br from-violet-900/40 to-gray-900 border-violet-500 shadow-md shadow-violet-500/20"
                      : "bg-gradient-to-br from-brand-900/40 to-gray-900 border-brand-500 shadow-md shadow-brand-500/20"
                    : t.is2D
                      ? "bg-gradient-to-br from-gray-900 to-gray-800 border-white/10 hover:from-violet-900/30 hover:to-gray-900 hover:border-violet-500/50 hover:shadow-lg hover:shadow-violet-500/10 hover:scale-[1.02]"
                      : "bg-gradient-to-br from-gray-900 to-gray-800 border-white/10 hover:from-brand-900/30 hover:to-gray-900 hover:border-brand-500/50 hover:shadow-lg hover:shadow-brand-500/10 hover:scale-[1.02]"
                  }
                `}
              >
                <Icon
                  size={36}
                  className={`mb-2 ${
                    isSelected
                      ? t.is2D ? "text-violet-400" : "text-brand-400"
                      : "text-gray-400"
                  }`}
                />
                <div className="text-sm font-semibold text-white">{t.label}</div>
                <div className="text-xs text-gray-400 mt-0.5">{t.description}</div>
                {/* accent dot */}
                <div className={`inline-block mt-2 h-1 w-6 rounded-full ${
                  isSelected
                    ? t.is2D ? "bg-violet-500" : "bg-brand-500"
                    : "bg-transparent"
                }`} />
              </button>
            );
            void accent; // avoid unused var
          })}
        </div>
      </div>

      {/* Post-procesado group */}
      <div>
        <h3 className="text-xs font-semibold uppercase tracking-widest text-gray-500 mb-3">
          Post-procesado
        </h3>
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
          {postprocessTypes.map((t) => {
            const Icon = t.icon;
            const isSelected = jobType === t.value;
            return (
              <button
                key={t.value}
                type="button"
                onClick={() => selectJobType(t.value)}
                className={`
                  p-4 rounded-xl border text-left transition-all duration-200
                  backdrop-blur-md
                  ${isSelected
                    ? t.is2D
                      ? "bg-gradient-to-br from-violet-900/40 to-gray-900 border-violet-500 shadow-md shadow-violet-500/20"
                      : "bg-gradient-to-br from-brand-900/40 to-gray-900 border-brand-500 shadow-md shadow-brand-500/20"
                    : t.is2D
                      ? "bg-gradient-to-br from-gray-900 to-gray-800 border-white/10 hover:from-violet-900/30 hover:to-gray-900 hover:border-violet-500/50 hover:shadow-lg hover:shadow-violet-500/10 hover:scale-[1.02]"
                      : "bg-gradient-to-br from-gray-900 to-gray-800 border-white/10 hover:from-brand-900/30 hover:to-gray-900 hover:border-brand-500/50 hover:shadow-lg hover:shadow-brand-500/10 hover:scale-[1.02]"
                  }
                `}
              >
                <Icon
                  size={32}
                  className={`mb-2 ${
                    isSelected
                      ? t.is2D ? "text-violet-400" : "text-brand-400"
                      : "text-gray-400"
                  }`}
                />
                <div className="text-sm font-semibold text-white">{t.label}</div>
                <div className="text-xs text-gray-400 mt-0.5 leading-tight">{t.description}</div>
              </button>
            );
          })}
        </div>
      </div>
    </div>
  );

  // ─── Step 2: Configure ──────────────────────────────────────────────────────
  const renderStep2 = () => (
    <div className="animate-slide-in space-y-5">
      {/* Selected type badge */}
      <div className="flex items-center gap-2">
        {(() => {
          const Icon = selectedTypeDef.icon;
          return (
            <div className={`p-2 rounded-lg ${selectedTypeDef.is2D ? "bg-violet-500/20" : "bg-brand-500/20"}`}>
              <Icon size={18} className={selectedTypeDef.is2D ? "text-violet-400" : "text-brand-400"} />
            </div>
          );
        })()}
        <div>
          <div className="text-sm font-semibold text-white">{selectedTypeDef.label}</div>
          <div className="text-xs text-gray-500">{selectedTypeDef.description}</div>
        </div>
      </div>

      {/* File upload for animate/refine/skin */}
      {selectedTypeDef.needsFile && (
        <div>
          <label className="block text-sm text-gray-300 mb-2">Archivo GLB de entrada</label>
          <div
            onClick={() => fileRef.current?.click()}
            className={`border-2 border-dashed rounded-xl p-5 text-center cursor-pointer transition-all duration-200 ${
              file
                ? "border-brand-500 bg-brand-600/10"
                : "border-gray-700 hover:border-gray-500 hover:bg-white/5"
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
                <p className="text-sm text-white font-medium">{file.name}</p>
                <p className="text-xs text-gray-400 mt-1">{(file.size / 1024 / 1024).toFixed(1)} MB</p>
              </div>
            ) : (
              <div className="space-y-1">
                <p className="text-sm text-gray-400">Click para seleccionar archivo</p>
                <p className="text-xs text-gray-600">.glb</p>
              </div>
            )}
          </div>
        </div>
      )}

      {/* Source job ID for animate_2d */}
      {jobType === "animate_2d" && (
        <div>
          <label className="block text-sm text-gray-300 mb-2">
            ID del job <span className="text-violet-400 font-medium">Generar 2D</span> de origen
          </label>
          <input
            type="number"
            value={sourceJobId}
            onChange={(e) => setSourceJobId(e.target.value)}
            placeholder="Ej: 42"
            min={1}
            className="w-full bg-gray-900 border border-gray-700 rounded-xl px-4 py-2.5 text-white placeholder-gray-500 focus:outline-none focus:ring-2 focus:ring-violet-500 focus:border-transparent"
            disabled={loading}
          />
          <p className="text-xs text-gray-500 mt-1">
            Puedes encontrar el ID en la URL del job completado: /job?id=42
          </p>
        </div>
      )}

      {/* Animation quality presets for animate_2d */}
      {jobType === "animate_2d" && (
        <div className="space-y-3">
          <label className="block text-sm font-medium text-gray-300">Calidad de animación</label>
          <div className="grid grid-cols-2 gap-2">
            {ANIM_PRESETS.map((preset) => {
              const presetVram = estimateVramMb(preset.frames, preset.res);
              const isAvailable = gpuTotalMb == null || presetVram <= gpuTotalMb;
              const isSelected =
                animationParams.num_frames === preset.frames &&
                animationParams.anim_inference_steps === preset.steps &&
                animationParams.anim_resolution === preset.res;
              return (
                <button
                  key={preset.key}
                  type="button"
                  disabled={!isAvailable || loading}
                  onClick={() => {
                    setAnimationParams((prev) => ({
                      ...prev,
                      num_frames: preset.frames,
                      anim_inference_steps: preset.steps,
                      anim_guidance_scale: preset.guidance,
                      anim_resolution: preset.res,
                    }));
                  }}
                  className={`p-3 rounded-xl border text-left transition-all duration-200 ${
                    !isAvailable
                      ? "bg-gray-900/30 border-gray-800 text-gray-600 cursor-not-allowed opacity-50"
                      : isSelected
                        ? "bg-gradient-to-br from-violet-900/40 to-gray-900 border-violet-500 shadow-md shadow-violet-500/20 text-white"
                        : "bg-gradient-to-br from-gray-900 to-gray-800 border-white/10 text-gray-400 hover:from-violet-900/20 hover:to-gray-900 hover:border-violet-500/40 hover:text-white hover:scale-[1.02]"
                  }`}
                  title={!isAvailable ? `Necesita ~${(presetVram / 1024).toFixed(1)} GB VRAM` : ""}
                >
                  <div className="text-sm font-semibold">{preset.label}</div>
                  <div className="text-xs mt-0.5 opacity-70">{preset.desc}</div>
                  {!isAvailable && (
                    <div className="text-xs mt-1 text-red-400">VRAM insuficiente</div>
                  )}
                </button>
              );
            })}
          </div>

          {/* VRAM widget always visible for animate_2d */}
          <VramWidget
            numFrames={animationParams.num_frames}
            steps={animationParams.anim_inference_steps}
            resolution={animationParams.anim_resolution}
            gpuName={gpuName}
            gpuTotalMb={gpuTotalMb}
          />
        </div>
      )}

      {/* Style picker for generate_2d */}
      {jobType === "generate_2d" && (
        <div className="space-y-2">
          <label className="block text-sm font-medium text-gray-300">Estilo artístico</label>
          <div className="grid grid-cols-3 gap-2">
            {STYLE_OPTIONS.map((opt) => {
              const isSelected = style2D === opt.value;
              return (
                <button
                  key={opt.value}
                  type="button"
                  onClick={() => setStyle2D(opt.value)}
                  disabled={loading}
                  className={`p-3 rounded-xl border text-left transition-all duration-200 ${
                    isSelected
                      ? "bg-gradient-to-br from-violet-900/40 to-gray-900 border-violet-500 shadow-md shadow-violet-500/20"
                      : "bg-gradient-to-br from-gray-900 to-gray-800 border-white/10 hover:from-violet-900/20 hover:to-gray-900 hover:border-violet-500/40 hover:scale-[1.02]"
                  }`}
                >
                  <div className="text-lg mb-1">{STYLE_ICONS[opt.value] ?? "🎨"}</div>
                  <div className={`text-xs font-medium ${isSelected ? "text-violet-300" : "text-gray-300"}`}>
                    {opt.label}
                  </div>
                </button>
              );
            })}
          </div>
        </div>
      )}

      {/* Prompt textarea */}
      <div>
        <label htmlFor="wizard-prompt" className="block text-sm font-medium text-gray-300 mb-2">
          {jobType === "animate"     ? "Describe la animación" :
           jobType === "animate_2d"  ? "Describe la animación (idle, walk, attack, dance…)" :
           jobType === "refine"      ? "Instrucciones de mejora" :
           jobType === "scene"       ? "Describe el escenario" :
           jobType === "skin"        ? "Describe la textura / materiales" :
           jobType === "generate_2d" ? "Describe tu personaje 2D" :
           "Describe tu modelo 3D"}
        </label>
        <textarea
          id="wizard-prompt"
          value={prompt}
          onChange={(e) => setPrompt(e.target.value)}
          placeholder={promptPlaceholders[jobType]}
          rows={3}
          className={`w-full bg-gray-900 border rounded-xl px-4 py-3 text-white placeholder-gray-500 focus:outline-none focus:ring-2 focus:border-transparent resize-none transition-colors ${
            selectedTypeDef.is2D
              ? "border-gray-700 focus:ring-violet-500"
              : "border-gray-700 focus:ring-brand-500"
          }`}
          disabled={loading}
          maxLength={2000}
        />
      </div>

      {/* Navigation */}
      <div className="flex gap-3 pt-1">
        <button
          type="button"
          onClick={goBack}
          className="px-5 py-2.5 text-sm text-gray-400 border border-gray-700 rounded-xl hover:text-white hover:border-gray-500 transition-colors"
        >
          ← Atrás
        </button>
        <button
          type="button"
          onClick={goNext}
          disabled={!canProceedToStep3}
          className={`flex-1 py-2.5 text-sm font-semibold rounded-xl transition-all duration-200 ${
            canProceedToStep3
              ? selectedTypeDef.is2D
                ? "bg-violet-600 hover:bg-violet-700 text-white"
                : "bg-brand-600 hover:bg-brand-700 text-white"
              : "bg-gray-800 text-gray-500 cursor-not-allowed"
          }`}
        >
          Revisar y generar →
        </button>
      </div>
    </div>
  );

  // ─── Step 3: Review & Advanced ──────────────────────────────────────────────
  const renderStep3 = () => (
    <div className="animate-slide-in space-y-5">
      {/* Summary card */}
      <div className="glass-card p-4 space-y-3">
        <h3 className="text-sm font-semibold text-gray-300">Resumen</h3>
        <div className="flex items-center gap-2">
          {(() => {
            const Icon = selectedTypeDef.icon;
            return (
              <div className={`p-1.5 rounded-lg ${selectedTypeDef.is2D ? "bg-violet-500/20" : "bg-brand-500/20"}`}>
                <Icon size={14} className={selectedTypeDef.is2D ? "text-violet-400" : "text-brand-400"} />
              </div>
            );
          })()}
          <span className="text-sm text-white font-medium">{selectedTypeDef.label}</span>
          {jobType === "generate_2d" && (
            <span className="text-xs text-violet-400 bg-violet-900/30 px-2 py-0.5 rounded-full">
              {STYLE_OPTIONS.find((s) => s.value === style2D)?.label ?? style2D}
            </span>
          )}
          {jobType === "animate_2d" && (
            <span className="text-xs text-violet-400 bg-violet-900/30 px-2 py-0.5 rounded-full">
              {ANIM_PRESETS.find(
                (p) =>
                  p.frames === animationParams.num_frames &&
                  p.steps === animationParams.anim_inference_steps &&
                  p.res === animationParams.anim_resolution,
              )?.label ?? "Custom"}
            </span>
          )}
        </div>
        <p className="text-sm text-gray-300 bg-gray-900/60 rounded-lg px-3 py-2 border border-gray-800">
          {prompt}
        </p>
        {jobType === "animate_2d" && sourceJobId && (
          <div className="text-xs text-gray-500">
            Source job ID: <span className="text-violet-400 font-mono">#{sourceJobId}</span>
          </div>
        )}
        {file && (
          <div className="text-xs text-gray-500">
            Archivo: <span className="text-white">{file.name}</span>
          </div>
        )}
      </div>

      {/* VRAM widget (always visible for animate_2d in step 3) */}
      {jobType === "animate_2d" && (
        <VramWidget
          numFrames={animationParams.num_frames}
          steps={animationParams.anim_inference_steps}
          resolution={animationParams.anim_resolution}
          gpuName={gpuName}
          gpuTotalMb={gpuTotalMb}
        />
      )}

      {/* Collapsible advanced options */}
      <div className={`rounded-xl border overflow-hidden transition-colors ${
        selectedTypeDef.is2D ? "border-violet-800/60" : "border-gray-700"
      }`}>
        <button
          type="button"
          onClick={() => setShowAdvanced(!showAdvanced)}
          className={`w-full flex items-center justify-between px-4 py-3 text-sm font-medium transition-colors ${
            selectedTypeDef.is2D
              ? "text-violet-300 hover:text-white hover:bg-violet-900/20"
              : "text-gray-300 hover:text-white hover:bg-gray-800"
          }`}
        >
          <span>Opciones avanzadas</span>
          {showAdvanced ? <ChevronUp size={16} /> : <ChevronDown size={16} />}
        </button>

        {showAdvanced && (
          <div className={`px-4 pb-4 space-y-4 border-t ${
            selectedTypeDef.is2D ? "border-violet-800/40 bg-gray-900/40" : "border-gray-800 bg-gray-900/40"
          }`}>

            {/* Common: negative prompt + seed */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4 pt-3">
              <div>
                <label className="block text-xs text-gray-400 mb-1">Prompt negativo</label>
                <input
                  type="text"
                  value={negativePrompt}
                  onChange={(e) => setNegativePrompt(e.target.value)}
                  placeholder="Cosas que evitar..."
                  className="w-full bg-gray-800 border border-gray-700 rounded-lg px-3 py-2 text-sm text-white placeholder-gray-500 focus:outline-none focus:ring-1 focus:ring-brand-500"
                  disabled={loading}
                />
              </div>
              <div>
                <label className="block text-xs text-gray-400 mb-1">Seed</label>
                <input
                  type="number"
                  value={seed}
                  onChange={(e) => setSeed(e.target.value)}
                  placeholder="Aleatorio"
                  min={0}
                  className="w-full bg-gray-800 border border-gray-700 rounded-lg px-3 py-2 text-sm text-white placeholder-gray-500 focus:outline-none focus:ring-1 focus:ring-brand-500"
                  disabled={loading}
                />
              </div>
            </div>

            {/* generate/scene: steps + guidance */}
            {(jobType === "generate" || jobType === "scene" || jobType === "generate_2d") && (
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div>
                  <label className="block text-xs text-gray-400 mb-1">Steps: <span className="text-white">{numSteps}</span></label>
                  <input
                    type="range"
                    value={numSteps}
                    onChange={(e) => setNumSteps(parseInt(e.target.value))}
                    min={1}
                    max={100}
                    className="w-full accent-brand-500"
                    disabled={loading}
                  />
                </div>
                <div>
                  <label className="block text-xs text-gray-400 mb-1">Guidance: <span className="text-white">{guidanceScale}</span></label>
                  <input
                    type="range"
                    value={guidanceScale}
                    onChange={(e) => setGuidanceScale(parseFloat(e.target.value))}
                    min={1}
                    max={20}
                    step={0.5}
                    className="w-full accent-brand-500"
                    disabled={loading}
                  />
                </div>
              </div>
            )}

            {/* animate_2d: detailed animation controls */}
            {jobType === "animate_2d" && (
              <div className="space-y-4">
                <button
                  type="button"
                  onClick={() => setShowAnimOptions(!showAnimOptions)}
                  className="text-xs text-violet-400 hover:text-violet-300 transition-colors"
                >
                  {showAnimOptions ? "- Ocultar controles de animación" : "+ Controles de animación avanzados"}
                </button>

                {showAnimOptions && (
                  <div className="space-y-4">
                    {/* frames + resolution */}
                    <div className="grid grid-cols-2 gap-4">
                      <div>
                        <label className="block text-xs text-gray-400 mb-1">Fotogramas</label>
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

                    {/* guidance_scale slider */}
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

                    {/* inference_steps slider */}
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

                    {/* enhance controls */}
                    <div className="space-y-3 pt-1 border-t border-gray-800">
                      <div className="flex items-center gap-2 pt-2">
                        <input
                          type="checkbox"
                          id="wizard-enhance_animation"
                          checked={animationParams.enhance_animation}
                          onChange={(e) => updateAnim("enhance_animation", e.target.checked)}
                          disabled={loading}
                          className="w-4 h-4 accent-violet-500 disabled:opacity-50"
                        />
                        <label htmlFor="wizard-enhance_animation" className="text-sm text-gray-300 cursor-pointer">
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
                  </div>
                )}
              </div>
            )}
          </div>
        )}
      </div>

      {error && (
        <div className="text-red-400 text-sm bg-red-900/20 border border-red-800 rounded-xl px-3 py-2">
          {error}
        </div>
      )}

      {/* Navigation */}
      <div className="flex gap-3 pt-1">
        <button
          type="button"
          onClick={goBack}
          className="px-5 py-2.5 text-sm text-gray-400 border border-gray-700 rounded-xl hover:text-white hover:border-gray-500 transition-colors"
          disabled={loading}
        >
          ← Atrás
        </button>
        <button
          type="submit"
          disabled={loading || !prompt.trim() || (selectedTypeDef.needsFile && !file) || (jobType === "animate_2d" && !sourceJobId.trim())}
          className={`flex-1 py-2.5 text-sm font-semibold rounded-xl transition-all duration-200 ${
            selectedTypeDef.is2D
              ? "bg-violet-600 hover:bg-violet-700 disabled:bg-gray-700"
              : "bg-brand-600 hover:bg-brand-700 disabled:bg-gray-700"
          } disabled:cursor-not-allowed text-white`}
        >
          {loading ? (
            <span className="flex items-center justify-center gap-2">
              <svg className="w-4 h-4 animate-spin" fill="none" viewBox="0 0 24 24">
                <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
              </svg>
              Procesando...
            </span>
          ) : (
            `Generar — ${selectedTypeDef.label}`
          )}
        </button>
      </div>
    </div>
  );

  return (
    <form onSubmit={handleSubmit} className="space-y-2">
      <StepIndicator currentStep={step} />

      {step === 1 && renderStep1()}
      {step === 2 && renderStep2()}
      {step === 3 && renderStep3()}
    </form>
  );
}
