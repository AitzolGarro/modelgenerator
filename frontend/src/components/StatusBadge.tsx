"use client";

import { STATUS_LABELS, STATUS_COLORS, type JobStatus } from "@/types/job";

export default function StatusBadge({ status }: { status: JobStatus }) {
  return (
    <span className="inline-flex items-center gap-1.5 text-xs font-medium">
      <span className={`w-2 h-2 rounded-full ${STATUS_COLORS[status]}`} />
      {STATUS_LABELS[status]}
    </span>
  );
}
