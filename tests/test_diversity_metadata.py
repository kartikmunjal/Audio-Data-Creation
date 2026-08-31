import math

import pandas as pd

from audio_curation.diversity import DiversityAnalyzer


def test_locale_is_not_mislabeled_as_accent():
    report = DiversityAnalyzer(pd.DataFrame({"locale": ["en", "en"]})).report()
    assert report["accent_distribution"] == {}
    assert math.isnan(report["accent_entropy"])
    assert report["locale_distribution"]["count"] == {"en": 2}
    assert "accent" not in report["diversity_scores"]
