from __future__ import annotations

import os
from typing import Any


def emit_run_span(run_id: str, attributes: dict[str, Any]) -> bool:
    endpoint = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT")
    if not endpoint: return False
    try:
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor
        provider = TracerProvider(resource=Resource.create({"service.name": "tifa"})); processor = BatchSpanProcessor(OTLPSpanExporter(endpoint=endpoint.rstrip("/") + "/v1/traces")); provider.add_span_processor(processor)
        tracer = provider.get_tracer("tifa.runtime")
        with tracer.start_as_current_span("tifa.run") as span:
            span.set_attribute("tifa.run_id", run_id)
            for key, value in attributes.items():
                if isinstance(value, (str, bool, int, float)): span.set_attribute(f"tifa.{key}", value)
        provider.force_flush(); provider.shutdown(); return True
    except (ImportError, RuntimeError, OSError): return False
