"use client";

import Link from "next/link";
import type { Job } from "@/types/job";
import { JOB_TYPE_LABELS } from "@/types/job";
import StatusBadge from "./StatusBadge";

export default function JobCard({ job }: { job: Job }) {
  const timeAgo = getTimeAgo(job.created_at);

  return (
    <Link
      href={`/job?id=${job.id}`}
      className="block p-4 bg-gray-900 border border-gray-800 rounded-lg hover:border-gray-600 transition-colors"
    >
      <div className="flex items-start justify-between gap-3">
        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-2 mb-1">
            <span className="text-[10px] uppercase tracking-wider text-gray-500 bg-gray-800 px-1.5 py-0.5 rounded">
              {JOB_TYPE_LABELS[job.job_type] || job.job_type}
            </span>
          </div>
          <p className="text-sm text-white truncate">{job.prompt}</p>
          <div className="flex items-center gap-3 mt-2">
            <StatusBadge status={job.status} />
            <span className="text-xs text-gray-500">{timeAgo}</span>
          </div>
        </div>

        {job.image_url && (
          <div className="w-12 h-12 rounded overflow-hidden flex-shrink-0 bg-gray-800">
            <img src={job.image_url} alt="" className="w-full h-full object-cover" />
          </div>
        )}
      </div>

      {job.error_message && (
        <p className="text-xs text-red-400 mt-2 truncate">{job.error_message}</p>
      )}
    </Link>
  );
}

function getTimeAgo(dateStr: string): string {
  const seconds = Math.floor((Date.now() - new Date(dateStr).getTime()) / 1000);
  if (seconds < 60) return "ahora";
  if (seconds < 3600) return `hace ${Math.floor(seconds / 60)}m`;
  if (seconds < 86400) return `hace ${Math.floor(seconds / 3600)}h`;
  return `hace ${Math.floor(seconds / 86400)}d`;
}
