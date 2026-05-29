import { useState } from "react";
import { predictApplication } from "./api/client";
import ScoringForm from "./components/ScoringForm";
import ResultPanel from "./components/ResultPanel";
import type { PredictRequest, PredictResponse } from "./types/api";
import "./App.css";

export default function App() {
  const [result, setResult] = useState<PredictResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  const handleSubmit = async (payload: PredictRequest) => {
    setLoading(true);
    setError(null);

    try {
      const response = await predictApplication(payload);
      setResult(response);
    } catch (err) {
      setResult(null);
      setError(err instanceof Error ? err.message : "Неизвестная ошибка");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="layout">
      <header className="header">
        <h1>Кредитный скоринг</h1>
        <p>Веб-приложение для оценки заявок с объяснимым решением</p>
      </header>

      <main className="grid">
        <ScoringForm onSubmit={handleSubmit} loading={loading} />
        <ResultPanel result={result} error={error} />
      </main>
    </div>
  );
}
