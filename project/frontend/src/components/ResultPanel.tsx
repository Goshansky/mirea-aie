import type { Decision, PredictResponse } from "../types/api";

const DECISION_LABELS: Record<Decision, string> = {
  APPROVE: "Одобрить",
  REJECT: "Отклонить",
  REVIEW: "На ручную проверку",
};

interface ResultPanelProps {
  result: PredictResponse | null;
  error: string | null;
}

/** Панель результата скоринга. */
export default function ResultPanel({ result, error }: ResultPanelProps) {
  if (error) {
    return (
      <section className="card result result--error">
        <h2>Ошибка</h2>
        <p>{error}</p>
      </section>
    );
  }

  if (!result) {
    return (
      <section className="card result result--empty">
        <h2>Результат</h2>
        <p>Заполните форму и нажмите «Оценить».</p>
      </section>
    );
  }

  const decisionClass = `decision decision--${result.decision.toLowerCase()}`;
  const percent = (result.probability * 100).toFixed(1);
  const decisionLabel = DECISION_LABELS[result.decision] ?? result.decision;

  return (
    <section className="card result">
      <h2>Результат</h2>

      <div className={decisionClass}>{decisionLabel}</div>

      <p className="probability">
        Вероятность дефолта: <strong>{percent}%</strong>
      </p>

      <div className="reasons">
        <h3>Причины решения</h3>
        <ul>
          {result.reasons.map((reason) => (
            <li key={reason}>{reason}</li>
          ))}
        </ul>
      </div>
    </section>
  );
}
