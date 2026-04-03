"use client";

import { useEffect, useState } from "react";
import { listJobs } from "@/lib/api";
import type { Job, JobStatus } from "@/types/job";
import JobCard from "@/components/JobCard";

const STATUSES: { value: string; label: string }[] = [
  { value: "", label: "Todos" },
  { value: "pending", label: "Pendiente" },
  { value: "completed", label: "Completado" },
  { value: "failed", label: "Error" },
];

export default function JobsPage() {
  const [jobs, setJobs] = useState<Job[]>([]);
  const [total, setTotal] = useState(0);
  const [page, setPage] = useState(1);
  const [statusFilter, setStatusFilter] = useState("");
  const [loading, setLoading] = useState(true);

  const pageSize = 20;

  useEffect(() => {
    let active = true;

    async function load() {
      setLoading(true);
      try {
        const data = await listJobs(page, pageSize, statusFilter || undefined);
        if (active) {
          setJobs(data.jobs);
          setTotal(data.total);
        }
      } catch {
        // API not available
      } finally {
        if (active) setLoading(false);
      }
    }

    load();

    const interval = setInterval(load, 5000);
    return () => {
      active = false;
      clearInterval(interval);
    };
  }, [page, statusFilter]);

  const totalPages = Math.ceil(total / pageSize);

  return (
    <div className="max-w-3xl mx-auto">
      <div className="flex items-center justify-between mb-6">
        <h1 className="text-2xl font-bold">Historial de Jobs</h1>
        <span className="text-sm text-gray-500">{total} jobs</span>
      </div>

      {/* Filters */}
      <div className="flex gap-2 mb-6">
        {STATUSES.map((s) => (
          <button
            key={s.value}
            onClick={() => {
              setStatusFilter(s.value);
              setPage(1);
            }}
            className={`px-3 py-1.5 text-xs rounded-full border transition-colors ${
              statusFilter === s.value
                ? "bg-brand-600 border-brand-500 text-white"
                : "bg-gray-900 border-gray-700 text-gray-400 hover:border-gray-500"
            }`}
          >
            {s.label}
          </button>
        ))}
      </div>

      {/* Job list */}
      {loading ? (
        <div className="space-y-3">
          {[...Array(5)].map((_, i) => (
            <div
              key={i}
              className="h-20 bg-gray-900 rounded-lg animate-pulse border border-gray-800"
            />
          ))}
        </div>
      ) : jobs.length === 0 ? (
        <p className="text-gray-500 text-center py-12">No hay jobs.</p>
      ) : (
        <div className="space-y-3">
          {jobs.map((job) => (
            <JobCard key={job.id} job={job} />
          ))}
        </div>
      )}

      {/* Pagination */}
      {totalPages > 1 && (
        <div className="flex items-center justify-center gap-2 mt-8">
          <button
            onClick={() => setPage(Math.max(1, page - 1))}
            disabled={page === 1}
            className="px-3 py-1.5 text-sm bg-gray-900 border border-gray-700 rounded disabled:opacity-30"
          >
            Anterior
          </button>
          <span className="text-sm text-gray-400">
            {page} / {totalPages}
          </span>
          <button
            onClick={() => setPage(Math.min(totalPages, page + 1))}
            disabled={page === totalPages}
            className="px-3 py-1.5 text-sm bg-gray-900 border border-gray-700 rounded disabled:opacity-30"
          >
            Siguiente
          </button>
        </div>
      )}
    </div>
  );
}
