import PromptForm from "@/components/PromptForm";
import RecentJobs from "@/components/RecentJobs";

export default function HomePage() {
  return (
    <div className="max-w-2xl mx-auto space-y-10">
      {/* Header */}
      <div className="text-center space-y-3">
        <h1 className="text-3xl font-bold">
          Genera modelos 3D desde texto
        </h1>
        <p className="text-gray-400">
          Describe lo que quieres crear y el pipeline lo convierte en un modelo
          3D listo para descargar.
        </p>
      </div>

      {/* Prompt form */}
      <div className="bg-gray-900/50 border border-gray-800 rounded-xl p-6">
        <PromptForm />
      </div>

      {/* Recent jobs */}
      <div>
        <h2 className="text-lg font-semibold mb-4">Jobs recientes</h2>
        <RecentJobs />
      </div>
    </div>
  );
}
