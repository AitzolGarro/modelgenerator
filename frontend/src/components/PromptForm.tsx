"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";
import { createJob } from "@/lib/api";
import type { JobCreatePayload } from "@/types/job";

export default function PromptForm() {
  const router = useRouter();
  const [prompt, setPrompt] = useState("");
  const [negativePrompt, setNegativePrompt] = useState("");
  const [numSteps, setNumSteps] = useState(30);
  const [guidanceScale, setGuidanceScale] = useState(7.5);
  const [seed, setSeed] = useState<string>("");
  const [showAdvanced, setShowAdvanced] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    if (!prompt.trim()) return;

    setLoading(true);
    setError(null);

    try {
      const payload: JobCreatePayload = {
        prompt: prompt.trim(),
        num_steps: numSteps,
        guidance_scale: guidanceScale,
      };

      if (negativePrompt.trim()) {
        payload.negative_prompt = negativePrompt.trim();
      }

      if (seed.trim()) {
        payload.seed = parseInt(seed, 10);
      }

      const job = await createJob(payload);
      router.push(`/job?id=${job.id}`);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Error creating job");
    } finally {
      setLoading(false);
    }
  }

  return (
    <form onSubmit={handleSubmit} className="space-y-4">
      {/* Main prompt */}
      <div>
        <label htmlFor="prompt" className="block text-sm font-medium text-gray-300 mb-1">
          Describe tu modelo 3D
        </label>
        <textarea
          id="prompt"
          value={prompt}
          onChange={(e) => setPrompt(e.target.value)}
          placeholder="Ej: A detailed medieval sword with ornate handle and gemstone"
          rows={3}
          className="w-full bg-gray-900 border border-gray-700 rounded-lg px-4 py-3 text-white placeholder-gray-500 focus:outline-none focus:ring-2 focus:ring-brand-500 focus:border-transparent resize-none"
          disabled={loading}
          maxLength={2000}
        />
      </div>

      {/* Advanced options toggle */}
      <button
        type="button"
        onClick={() => setShowAdvanced(!showAdvanced)}
        className="text-sm text-gray-400 hover:text-white transition-colors"
      >
        {showAdvanced ? "- Ocultar opciones" : "+ Opciones avanzadas"}
      </button>

      {/* Advanced options */}
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
            <label className="block text-sm text-gray-400 mb-1">
              Seed (dejar vacio = aleatorio)
            </label>
            <input
              type="number"
              value={seed}
              onChange={(e) => setSeed(e.target.value)}
              placeholder="42"
              min={0}
              className="w-full bg-gray-900 border border-gray-700 rounded px-3 py-2 text-sm text-white placeholder-gray-500 focus:outline-none focus:ring-1 focus:ring-brand-500"
              disabled={loading}
            />
          </div>
          <div>
            <label className="block text-sm text-gray-400 mb-1">
              Steps: {numSteps}
            </label>
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
            <label className="block text-sm text-gray-400 mb-1">
              Guidance Scale: {guidanceScale}
            </label>
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

      {/* Error */}
      {error && (
        <div className="text-red-400 text-sm bg-red-900/20 border border-red-800 rounded px-3 py-2">
          {error}
        </div>
      )}

      {/* Submit */}
      <button
        type="submit"
        disabled={loading || !prompt.trim()}
        className="w-full bg-brand-600 hover:bg-brand-700 disabled:bg-gray-700 disabled:cursor-not-allowed text-white font-medium py-3 px-6 rounded-lg transition-colors"
      >
        {loading ? "Creando job..." : "Generar modelo 3D"}
      </button>
    </form>
  );
}
