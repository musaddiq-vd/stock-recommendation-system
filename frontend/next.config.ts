import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  // self-contained server bundle for the Docker image
  output: "standalone",
  poweredByHeader: false,
  // no next/image in this app: skip the image optimizer (and its native sharp dependency) entirely
  images: { unoptimized: true },
  async headers() {
    return [
      {
        source: "/:path*",
        headers: [
          { key: "X-Frame-Options", value: "DENY" },
          { key: "X-Content-Type-Options", value: "nosniff" },
          { key: "Referrer-Policy", value: "strict-origin-when-cross-origin" },
        ],
      },
    ];
  },
};

export default nextConfig;
