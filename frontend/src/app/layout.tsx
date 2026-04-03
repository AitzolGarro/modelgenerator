import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "ModelGenerator - Text to 3D",
  description: "Generate, animate, refine 3D models and create scenes from text prompts",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body className="min-h-screen bg-gray-950 text-gray-100">
        <nav className="border-b border-gray-800 bg-gray-950/80 backdrop-blur sticky top-0 z-50">
          <div className="max-w-7xl mx-auto px-4 h-14 flex items-center justify-between">
            <a href="/" className="flex items-center gap-2 font-semibold text-lg">
              <span className="text-brand-400">&#9651;</span>
              <span>ModelGenerator</span>
            </a>
            <div className="flex items-center gap-6 text-sm text-gray-400">
              <a href="/" className="hover:text-white transition-colors">Crear</a>
              <a href="/jobs" className="hover:text-white transition-colors">Historial</a>
            </div>
          </div>
        </nav>
        <main className="max-w-7xl mx-auto px-4 py-8">{children}</main>
      </body>
    </html>
  );
}
