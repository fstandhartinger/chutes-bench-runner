"""Tests for ArtificialAnalysisService parsing logic."""

from app.services.artificial_analysis_service import ArtificialAnalysisService


def test_extract_model_payload_prefers_metrics_blob() -> None:
    """
    AA /models/<slug> pages can contain multiple JSON blobs that repeat the same slug.

    We want the blob that includes benchmark/metric keys (e.g. aime25, gpqa, etc), not the
    smaller model-card blob that only includes metadata.
    """
    service = ArtificialAnalysisService()
    decoded = (
        '{"id":"1","slug":"glm-4-7","name":"GLM-4.7","shortName":"GLM-4.7"}'
        ' some intermediate text '
        '{"id":"2","slug":"glm-4-7","name":"GLM-4.7 (Reasoning)","aime25":0.95,"gpqa":0.859,'
        '"model_url":"/models/glm-4-7","hosts_url":"/models/glm-4-7/providers","short_name":"GLM-4.7"}'
    )

    payload = service._extract_model_payload(decoded, "glm-4-7")
    assert payload is not None
    assert payload.get("slug") == "glm-4-7"
    assert "aime25" in payload

