"""
Pipeline Module

End-to-end data pipeline for CORC-NAH project.

Production pipeline: UnifiedPipeline (from unify_v2)
- Progress bars, metadata tracking, graceful error handling
- Use for production runs

Legacy pipeline: src.pipeline.unify.UnifiedPipeline
- Lightweight version without progress bars
- Used in integration tests for simplicity
"""

from src.pipeline.unify_v2 import UnifiedPipeline

__all__ = ["UnifiedPipeline"]
