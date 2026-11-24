import { NextResponse } from "next/server";

function getBackendUrl(): string {
  return process.env.SOILVISION_BACKEND_URL || "http://127.0.0.1:8000";
}

export async function GET() {
  try {
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), 15000);
    const res = await fetch(new URL("/health", getBackendUrl()), {
      cache: "no-store",
      signal: controller.signal,
    });
    clearTimeout(timer);
    const data = await res.json();
    return NextResponse.json(data, { status: res.status });
  } catch {
    return NextResponse.json(
      {
        status: "offline",
        model_ready: false,
        model_version: null,
        error: "Unable to reach the SoilVision inference service.",
      },
      { status: 503 }
    );
  }
}
