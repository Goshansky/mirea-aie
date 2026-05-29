import type { PredictRequest, PredictResponse } from "../types/api";

/** Базовый URL API: env или proxy Vite (/api). */
const API_BASE = import.meta.env.VITE_API_URL?.replace(/\/$/, "") ?? "/api";

interface ApiErrorBody {
  detail?: string | unknown;
}

/** Отправляет заявку на скоринг. */
export async function predictApplication(payload: PredictRequest): Promise<PredictResponse> {
  const response = await fetch(`${API_BASE}/predict`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });

  if (!response.ok) {
    let detail = `Ошибка API (${response.status})`;
    try {
      const body = (await response.json()) as ApiErrorBody;
      if (body.detail !== undefined) {
        detail = typeof body.detail === "string" ? body.detail : JSON.stringify(body.detail);
      }
    } catch {
      // оставляем сообщение по умолчанию
    }
    throw new Error(detail);
  }

  return (await response.json()) as PredictResponse;
}
