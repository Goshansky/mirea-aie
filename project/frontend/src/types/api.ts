/** Решение по кредитной заявке. */
export type Decision = "APPROVE" | "REJECT" | "REVIEW";

/** Тело запроса POST /predict. */
export interface PredictRequest {
  income: number;
  loan_amount: number;
  age: number;
  credit_history: number;
  debt_ratio: number;
  late_payments: number;
}

/** Ответ POST /predict. */
export interface PredictResponse {
  decision: Decision;
  probability: number;
  reasons: string[];
}
