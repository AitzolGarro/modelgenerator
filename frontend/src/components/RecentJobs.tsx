"use client";

import { useEffect, useState } from "react";
import { listJobs } from "@/lib/api";
import type { Job } from "@/types/job";
import JobCard from "./JobCard";

export default function RecentJobs() {
  const [jobs, setJobs] = useState<Job[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    let active = true;

    async function load() {
      try {
        const data = await listJobs(1, 5);
        if (active) setJobs(data.jobs);
      } catch {
        // Silently fail - API might not be running
      } finally {
        if (active) setLoading(false);
      }
    }

    load();

    // Poll for updates
    const interval = setInterval(load, 5000);
    return () => {
      active = false;
      clearInterval(interval);
    };
  }, []);

  if (loading) {
    return (
      <div className="space-y-3">
        {[...Array(3)].map((_, i) => (
          <div
            key={i}
            className="h-20 bg-gray-900 rounded-lg animate-pulse border border-gray-800"
          />
        ))}
      </div>
    );
  }

  if (jobs.length === 0) {
    return (
      <p className="text-gray-500 text-sm text-center py-8">
        No hay jobs todavia. Crea uno arriba.
      </p>
    );
  }

  return (
    <div className="space-y-3">
      {jobs.map((job) => (
        <JobCard key={job.id} job={job} />
      ))}
    </div>
  );
}
