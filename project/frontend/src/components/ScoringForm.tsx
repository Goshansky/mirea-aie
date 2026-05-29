import type { FormEvent } from "react";
import type { PredictRequest } from "../types/api";

const INITIAL = {
  income: "120000",
  loan_amount: "50000",
  age: "35",
  credit_history: "7",
  debt_ratio: "0.31",
  late_payments: "0",
} as const;

interface ScoringFormProps {
  onSubmit: (payload: PredictRequest) => void;
  loading: boolean;
}

/** Форма ввода параметров кредитной заявки. */
export default function ScoringForm({ onSubmit, loading }: ScoringFormProps) {
  const handleSubmit = (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    const form = new FormData(event.currentTarget);

    onSubmit({
      income: Number(form.get("income")),
      loan_amount: Number(form.get("loan_amount")),
      age: Number(form.get("age")),
      credit_history: Number(form.get("credit_history")),
      debt_ratio: Number(form.get("debt_ratio")),
      late_payments: Number(form.get("late_payments") ?? 0),
    });
  };

  return (
    <form className="card form" onSubmit={handleSubmit}>
      <h2>Параметры заявки</h2>

      <label>
        Доход (₽/мес)
        <input name="income" type="number" min="0" step="1000" defaultValue={INITIAL.income} required />
      </label>

      <label>
        Сумма кредита (₽)
        <input name="loan_amount" type="number" min="0" step="1000" defaultValue={INITIAL.loan_amount} required />
      </label>

      <label>
        Возраст
        <input name="age" type="number" min="18" max="100" defaultValue={INITIAL.age} required />
      </label>

      <label>
        Стаж кредитной истории (лет)
        <input name="credit_history" type="number" min="0" max="80" defaultValue={INITIAL.credit_history} required />
      </label>

      <label>
        Debt ratio (0–1)
        <input
          name="debt_ratio"
          type="number"
          min="0"
          max="1"
          step="0.01"
          defaultValue={INITIAL.debt_ratio}
          required
        />
      </label>

      <label>
        Просрочки
        <input name="late_payments" type="number" min="0" max="50" defaultValue={INITIAL.late_payments} />
      </label>

      <button type="submit" disabled={loading}>
        {loading ? "Оценка…" : "Оценить"}
      </button>
    </form>
  );
}
