from rim.core.modes import DEEP_MODE, FAST_MODE, get_mode_settings


def test_default_deep_mode_settings() -> None:
    settings = get_mode_settings("deep")
    assert settings == DEEP_MODE
    assert settings.max_depth >= FAST_MODE.max_depth
    assert settings.self_critique_pass is True
    assert settings.max_children_per_node > FAST_MODE.max_children_per_node
    assert settings.runtime_budget_sec > FAST_MODE.runtime_budget_sec
    assert settings.max_total_nodes > FAST_MODE.max_total_nodes


def test_fast_mode_settings() -> None:
    settings = get_mode_settings("fast")
    assert settings == FAST_MODE
    assert settings.self_critique_pass is False
    assert settings.confidence_stop_threshold > DEEP_MODE.confidence_stop_threshold
