"use client";

import { useState, useRef } from "react";
import { useRouter } from "next/navigation";
import { createJob, uploadAndCreateJob } from "@/lib/api";
import type { JobType, JobCreatePayload } from "@/types/job";

const JOB_TYPES: { value: JobType; label: string; description: string; needsFile: boolean }[] = [
  { value: "generate", label: "Generar 3D",   description: "Texto → Imagen → Modelo 3D",    needsFile: false },
  { value: "animate",  label: "Animar",        description: "GLB + prompt → GLB animado",     needsFile: true },
  { value: "refine",   label: "Mejorar",       description: "GLB → mas detalle y calidad",    needsFile: true },
  { value: "scene",    label: "Escenario",     description: "Texto → Entorno 3D completo",    needsFile: false },
  { value: "skin",     label: "Texturizar",    description: "GLB + prompt → GLB texturizado", needsFile: true },
];

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
  const [showAdvanced, setShowAdvanced] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const selectedType = JOB_TYPES.find(t => t.value === jobType)!;

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    if (!prompt.trim()) return;

    setLoading(true);
    setError(null);

    try {
      let job;

      if (selectedType.needsFile && file) {
        // Upload mode
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
    generate: "Ej: A detailed medieval sword with ornate handle and gemstone",
    animate: "Ej: walking cycle, arms swinging naturally",
    refine: "Ej: increase detail, smooth surface, fix normals",
    scene: "Ej: enchanted forest with glowing mushrooms and a small lake",
    skin: "Ej: weathered stone texture with mossy cracks, dark fantasy style",
  };

  return (
    <form onSubmit={handleSubmit} className="space-y-4">
      {/* Job type selector */}
      <div className="grid grid-cols-2 md:grid-cols-5 gap-2">
        {JOB_TYPES.map((t) => (
          <button
            key={t.value}
            type="button"
            onClick={() => { setJobType(t.value); setFile(null); }}
            className={`p-3 rounded-lg border text-left transition-all ${
              jobType === t.value
                ? "bg-brand-600/20 border-brand-500 text-white"
                : "bg-gray-900 border-gray-700 text-gray-400 hover:border-gray-500"
            }`}
          >
            <div className="text-sm font-medium">{t.label}</div>
            <div className="text-xs mt-0.5 opacity-70">{t.description}</div>
          </button>
        ))}
      </div>

      {/* File upload for animate/refine */}
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

      {/* Prompt */}
      <div>
        <label htmlFor="prompt" className="block text-sm font-medium text-gray-300 mb-1">
          {jobType === "animate" ? "Describe la animacion" :
           jobType === "refine" ? "Instrucciones de mejora" :
           jobType === "scene" ? "Describe el escenario" :
           jobType === "skin" ? "Describe la textura / materiales" :
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
        disabled={loading || !prompt.trim() || (selectedType.needsFile && !file)}
        className="w-full bg-brand-600 hover:bg-brand-700 disabled:bg-gray-700 disabled:cursor-not-allowed text-white font-medium py-3 px-6 rounded-lg transition-colors"
      >
        {loading ? "Procesando..." : selectedType.label}
      </button>
    </form>
  );
}
