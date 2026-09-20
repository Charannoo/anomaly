"""Export-tolerance tests (Phase 7)."""

import pytest

pytestmark = pytest.mark.skip(reason="ONNX/OpenVINO export lands in Phase 7.")


def test_placeholder():  # pragma: no cover
    pass
