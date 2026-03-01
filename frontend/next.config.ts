import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  reactStrictMode: true,
  // Expose the backend URL at build time if needed by Server Components
  // (NEXT_PUBLIC_ vars are already handled automatically)
};

export default nextConfig;
