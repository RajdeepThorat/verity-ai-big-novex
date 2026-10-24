import time
import threading
from unittest.mock import patch, MagicMock
import pytest
import torch
from PIL import Image

import verity.ai_detect as ai_detect


def test_torch_threads_configuration():
    """Torch threads should be set sensibly for CPU inference."""
    ai_detect._configure_torch()
    threads = torch.get_num_threads()
    assert 1 <= threads <= 4


def test_preload_model_starts_background_thread():
    """preload_model should spawn a daemon thread."""
    with patch.object(ai_detect, "_preload_worker", return_value=None):
        t = ai_detect.preload_model()
        assert isinstance(t, threading.Thread)
        assert t.daemon is True
        t.join(timeout=1.0)


def test_retry_with_backoff_not_cached_forever():
    """Load error should not be cached forever; it must retry after backoff interval."""
    ai_detect._classifier = None
    ai_detect._load_error = None
    ai_detect._last_attempt_time = 0.0
    ai_detect._backoff_seconds = 0.05

    calls = 0

    def mock_load():
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RuntimeError("Temporary network timeout")
        return MagicMock()

    with patch.object(ai_detect, "_load_pipeline", side_effect=mock_load):
        # First call fails
        clf1 = ai_detect._get_classifier()
        assert clf1 is None
        assert ai_detect._load_error is not None
        assert calls == 1

        # Call immediately during backoff window -> returns None without calling _load_pipeline
        clf_immediate = ai_detect._get_classifier()
        assert clf_immediate is None
        assert calls == 1

        # Wait for backoff interval to expire (0.05 * 2 = 0.10s)
        time.sleep(0.15)

        # Call after backoff window -> retries and succeeds
        clf2 = ai_detect._get_classifier()
        assert clf2 is not None
        assert calls == 2
        assert ai_detect._load_error is None

    # Clean up state
    ai_detect._classifier = None
    ai_detect._load_error = None


def test_ai_probability_and_analyze():
    """ai_probability and analyze_ai should correctly parse model output."""
    mock_pipeline = MagicMock()
    mock_pipeline.return_value = [{"label": "artificial", "score": 0.92}]

    with patch.object(ai_detect, "_get_classifier", return_value=mock_pipeline):
        im = Image.new("RGB", (64, 64), color="green")
        prob = ai_detect.ai_probability(im)
        assert prob == pytest.approx(0.92)

    import tempfile, os
    tmp = os.path.join(tempfile.mkdtemp(), "test.png")
    im.save(tmp)

    with patch.object(ai_detect, "ai_probability", return_value=0.92):
        res = ai_detect.analyze_ai(tmp)
        assert res["available"] is True
        assert res["score"] == 0.92
        assert any("strong AI" in f for f in res["flags"])
