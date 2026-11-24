import { NextResponse } from "next/server";

const BACKEND_URL = process.env.SOILVISION_BACKEND_URL || "http://127.0.0.1:8000";
const BACKEND_API_KEY = process.env.SOILVISION_BACKEND_API_KEY || "";

export const maxDuration = 60;

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
    if (BACKEND_API_KEY) {
      headers["X-API-Key"] = BACKEND_API_KEY;
    }

    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), 55000);
    const res = await fetch(`${BACKEND_URL.replace(/\/$/, "")}/predict`, {
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
