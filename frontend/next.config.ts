import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  // Static export: generates /out directory with pure HTML/JS/CSS
  // This gets served by FastAPI in production mode
  output: "export",

  // In dev mode, proxy API requests to the backend
  async rewrites() {
    return [
      {
        source: "/api/:path*",
        destination: `${process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000"}/api/:path*`,
      },
    ];
  },

  // Disable image optimization for static export
  images: {
    unoptimized: true,
  },
};

export default nextConfig;
