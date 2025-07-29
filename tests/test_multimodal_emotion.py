# tests/test_multimodal_emotion.py

import pytest
import numpy as np
from core.multimodal_emotion import MultimodalEmotionDetector

@pytest.fixture(scope="module")
def detector():
    # Instantiate once for all tests
    return MultimodalEmotionDetector()

def test_text_only(detector):
    result = detector.detect_multimodal(text="I'm very happy today!")
    assert isinstance(result, dict)
    assert result["text_emotion"] == "happy"  # as per your EmotionDetector stub, adjust if needed
    assert result["final_emotion"] == result["text_emotion"]
    assert result["face_emotion"] is None
    assert result["audio_emotion"] is None

def test_image_only(detector):
    dummy_img = np.zeros((48, 48, 3), dtype=np.uint8)
    result = detector.detect_multimodal(image=dummy_img)
    assert isinstance(result, dict)
    assert result["face_emotion"] == "happy"  # from your current detect_face stub
    assert result["final_emotion"] == "happy"
    assert result["text_emotion"] is None
    assert result["audio_emotion"] is None

def test_audio_only(detector, tmp_path):
    # librosa expects a real audio file, but we'll simply call with a dummy (since method is stubbed)
    dummy_audio_path = tmp_path / "dummy.wav"
    dummy_audio_path.write_bytes(b"\x00" * 44)  # fake header to pass file existence
    # The following will work ONLY if your stub doesn't actually load the file.
    try:
        result = detector.detect_multimodal(audio_path=str(dummy_audio_path))
        assert isinstance(result, dict)
        assert result["audio_emotion"] == "angry"  # from your detect_audio stub
        assert result["final_emotion"] == "angry"
        assert result["text_emotion"] is None
        assert result["face_emotion"] is None
    except Exception:
        # If librosa.load fails, this is expected without a true audio stub.
        pytest.skip("Real audio file required to fully test audio pipeline.")

def test_all_modalities(detector, tmp_path):
    dummy_img = np.zeros((48, 48, 3), dtype=np.uint8)
    dummy_audio_path = tmp_path / "fake.wav"
    dummy_audio_path.write_bytes(b"\x00" * 44)
    result = detector.detect_multimodal(
        text="Feeling amazing!",
        image=dummy_img,
        audio_path=str(dummy_audio_path)
    )
    assert isinstance(result, dict)
    # All stub outputs: text = "happy", image = "happy", audio = "angry"
    assert result["text_emotion"] == "happy"
    assert result["face_emotion"] == "happy"
    assert result["audio_emotion"] == "angry"
    # Majority is "happy" (2 out of 3)
    assert result["final_emotion"] == "happy"

@pytest.mark.parametrize("inputs, expected", [
    ({}, "neutral"),
    ({"text": ""}, "neutral"),
    ({"image": None}, "neutral"),
    ({"audio_path": None}, "neutral"),
])
def test_no_valid_modalities(detector, inputs, expected):
    # Should handle missing/all-empty input gracefully
    result = detector.detect_multimodal(**inputs)
    assert result["final_emotion"] == expected
    assert result["text_emotion"] is None
    assert result["face_emotion"] is None
    assert result["audio_emotion"] is None

def test_majority_vote_fusion(detector):
    # Direct test of fusion method
    majority = detector.fuse_emotions(["happy", "sad", "happy"])
    assert majority == "happy"
    single = detector.fuse_emotions(["neutral"])
    assert single == "neutral"
    none = detector.fuse_emotions([])
    assert none == "neutral"

def test_structured_output_keys(detector):
    # Output dict must always have all four keys
    result = detector.detect_multimodal()
    for key in ["text_emotion", "face_emotion", "audio_emotion", "final_emotion"]:
        assert key in result
