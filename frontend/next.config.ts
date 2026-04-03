import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  // Static export: generates /out directory with pure HTML/JS/CSS
  // This gets served by FastAPI in production mode
  output: "export",

  // Disable image optimization for static export
  images: {
    unoptimized: true,
  },
};

export default nextConfig;
