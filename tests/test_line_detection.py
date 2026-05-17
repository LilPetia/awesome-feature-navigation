import numpy as np

from awesome_feature_navigation.line_detection import resolve_hsv_ranges, resolve_target_color


def test_resolve_target_color_defaults_to_blue_for_missing_or_invalid_values() -> None:
    assert resolve_target_color({}) == 'blue'
    assert resolve_target_color({'target_color': 'not-a-color'}) == 'blue'


def test_resolve_hsv_ranges_uses_blue_preset_by_default() -> None:
    ranges = resolve_hsv_ranges({})

    assert len(ranges) == 1
    low, high = ranges[0]
    np.testing.assert_array_equal(low, np.array([95, 100, 60], dtype=np.uint8))
    np.testing.assert_array_equal(high, np.array([130, 255, 255], dtype=np.uint8))


def test_legacy_red_hsv_ranges_apply_only_when_red_is_selected() -> None:
    cfg = {
        'target_color': 'blue',
        'hsv_red1_low': [0, 1, 2],
        'hsv_red1_high': [3, 4, 5],
    }

    low, high = resolve_hsv_ranges(cfg)[0]

    np.testing.assert_array_equal(low, np.array([95, 100, 60], dtype=np.uint8))
    np.testing.assert_array_equal(high, np.array([130, 255, 255], dtype=np.uint8))


def test_legacy_red_hsv_ranges_are_used_for_explicit_red() -> None:
    cfg = {
        'target_color': 'red',
        'hsv_red1_low': [10, 20, 30],
        'hsv_red1_high': [1, 200, 250],
    }

    low, high = resolve_hsv_ranges(cfg)[0]

    np.testing.assert_array_equal(low, np.array([1, 20, 30], dtype=np.uint8))
    np.testing.assert_array_equal(high, np.array([10, 200, 250], dtype=np.uint8))
