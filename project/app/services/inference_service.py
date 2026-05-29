"""Сервис инференса кредитного скоринга."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import joblib
import pandas as pd

from app.core.business_rules import (
    adjust_probability_for_risk_factors,
    apply_favorable_rules,
    apply_hard_rules,
)
from app.core.config import Settings
from app.core.decision import resolve_decision
from app.models.predictor import LoadedArtifacts
from app.models.schemas import PredictRequest, PredictResponse
from app.services.explanation_service import explain_with_importance, explain_with_shap
from ml.data.features import enrich_application_features

logger = logging.getLogger(__name__)


class InferenceService:
    """Сервис, который загружает модель и выполняет предсказания."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.artifacts = self._load_artifacts()

    def _load_artifacts(self) -> LoadedArtifacts:
        """Загружает модель и metadata из директории артефактов."""
        artifacts_dir = self.settings.artifacts_dir
        model_path = artifacts_dir / self.settings.model_file
        metadata_path = artifacts_dir / self.settings.feature_metadata_file

        if not model_path.exists():
            raise FileNotFoundError(f"Не найден файл модели: {model_path}")
        pipeline = joblib.load(model_path)

        raw_metadata: dict = {}
        feature_names: list[str] = []
        feature_importance: dict[str, float] = {}
        if metadata_path.exists():
            raw_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            feature_names = [str(name) for name in raw_metadata.get("feature_names", [])]
            feature_importance = {
                str(key): float(value)
                for key, value in raw_metadata.get("feature_importance", {}).items()
            }

        if not feature_names:
            preprocessor = pipeline.named_steps.get("preprocessor")
            try:
                feature_names = [str(name) for name in preprocessor.get_feature_names_out().tolist()]
            except Exception:
                feature_names = []

        return LoadedArtifacts(
            pipeline=pipeline,
            feature_names=feature_names,
            feature_importance=feature_importance,
            artifacts_dir=artifacts_dir,
            raw_metadata=raw_metadata,
        )

    def predict(self, payload: PredictRequest) -> PredictResponse:
        """Выполняет скоринг заявки и формирует ответ."""
        favorable_rule = apply_favorable_rules(payload, self.settings)
        if favorable_rule.triggered and favorable_rule.decision is not None:
            logger.info(
                "favorable_rule_approve probability=%.4f reasons=%s",
                favorable_rule.probability,
                favorable_rule.reasons,
            )
            return PredictResponse(
                decision=favorable_rule.decision.value,
                probability=round(float(favorable_rule.probability), 4),
                reasons=list(favorable_rule.reasons),
            )

        hard_rule = apply_hard_rules(payload, self.settings)
        if hard_rule.triggered and hard_rule.decision is not None:
            logger.info(
                "hard_rule_reject probability=%.4f reasons=%s",
                hard_rule.probability,
                hard_rule.reasons,
            )
            return PredictResponse(
                decision=hard_rule.decision.value,
                probability=round(float(hard_rule.probability), 4),
                reasons=list(hard_rule.reasons),
            )

        request_df = enrich_application_features(pd.DataFrame([payload.model_dump()]))

        pipeline = self.artifacts.pipeline
        ml_probability = float(pipeline.predict_proba(request_df)[0][1])
        pd_probability, risk_adjustments = adjust_probability_for_risk_factors(
            ml_probability,
            payload,
            self.settings,
        )
        decision = resolve_decision(
            pd_probability=pd_probability,
            threshold_approve=self.settings.threshold_approve,
            threshold_reject=self.settings.threshold_reject,
        ).value

        preprocessor = pipeline.named_steps["preprocessor"]
        model = pipeline.named_steps["model"]
        transformed_row = preprocessor.transform(request_df)
        if hasattr(transformed_row, "toarray"):
            transformed_row = transformed_row.toarray()

        raw_values = payload.model_dump()

        reasons = explain_with_shap(
            model=model,
            transformed_row=transformed_row,
            feature_names=self.artifacts.feature_names,
            top_k=3,
            raw_values=raw_values,
        )
        if not reasons:
            reasons = explain_with_importance(
                transformed_row=transformed_row,
                feature_names=self.artifacts.feature_names,
                feature_importance=self.artifacts.feature_importance,
                top_k=3,
                raw_values=raw_values,
            )

        reasons = risk_adjustments + reasons
        reasons = reasons[:3]

        logger.info(
            "prediction_result decision=%s ml_pd=%.4f adjusted_pd=%.4f lates=%d",
            decision,
            ml_probability,
            pd_probability,
            payload.late_payments,
        )

        return PredictResponse(
            decision=decision,
            probability=round(pd_probability, 4),
            reasons=reasons,
        )
