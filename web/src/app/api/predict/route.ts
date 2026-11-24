import { NextResponse } from "next/server";

export const maxDuration = 120;

function getBackendUrl(): string {
  return process.env.SOILVISION_BACKEND_URL || "http://127.0.0.1:8000";
}

export async function POST(request: Request) {
  try {
    const formData = await request.formData();
    const files = formData.getAll("files");
    if (!files.length) {
      return NextResponse.json(
        { detail: "Please attach at least one soil photograph." },
        { status: 400 }
      );
    }

    const headers: Record<string, string> = {};
    const apiKey = process.env.SOILVISION_BACKEND_API_KEY || "";
    if (apiKey) {
      headers["X-API-Key"] = apiKey;
    }

    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), 115000);
    const res = await fetch(new URL("/predict", getBackendUrl()), {
      method: "POST",
      headers,
      body: formData,
      signal: controller.signal,
    });
    clearTimeout(timer);

    const text = await res.text();
    try {
      const json = JSON.parse(text);
      return NextResponse.json(json, { status: res.status });
    } catch {
      return NextResponse.json(
        { detail: "Inference service returned a non-JSON response." },
        { status: res.ok ? 502 : res.status }
      );
    }
  } catch (err) {
    if (err instanceof Error && err.name === "AbortError") {
      return NextResponse.json(
        { detail: "Soil analysis timed out while waiting for the inference service." },
        { status: 504 }
      );
    }
    return NextResponse.json(
      { detail: "Failed to connect to the SoilVision inference service." },
      { status: 503 }
    );
  }
}
