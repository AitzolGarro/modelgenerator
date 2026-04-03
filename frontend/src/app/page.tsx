import PromptForm from "@/components/PromptForm";
import RecentJobs from "@/components/RecentJobs";

export default function HomePage() {
  return (
    <div className="max-w-2xl mx-auto space-y-10">
      <div className="text-center space-y-3">
        <h1 className="text-3xl font-bold">ModelGenerator</h1>
        <p className="text-gray-400">
          Genera, anima, mejora modelos 3D y crea escenarios completos desde texto.
        </p>
      </div>

      <div className="bg-gray-900/50 border border-gray-800 rounded-xl p-6">
        <PromptForm />
      </div>

      <div>
        <h2 className="text-lg font-semibold mb-4">Jobs recientes</h2>
        <RecentJobs />
      </div>
    </div>
  );
}
