from rim.agents.spawner import build_spawn_plan


def test_spawn_plan_selects_roles_from_constraints_and_domain() -> None:
    payload = build_spawn_plan(
        mode="deep",
        domain="FinTech",
        constraints=[
            "Must satisfy security and privacy controls.",
            "Keep latency low at scale.",
            "Stay within budget limits.",
        ],
        memory_context=[],
    )
    roles = [item["role"] for item in payload["extra_critics"]]
    assert payload["selected_count"] >= 2
    assert "security" in roles
    assert "scalability" in roles or "finance" in roles


def test_spawn_plan_limits_specialists_in_fast_mode() -> None:
    payload = build_spawn_plan(
        mode="fast",
        domain="general",
        constraints=[
            "security and privacy",
            "cost optimization",
            "latency and scale",
        ],
        memory_context=[],
    )
    assert payload["selected_count"] <= 1


def test_spawn_plan_includes_scoring_and_match_metadata() -> None:
    payload = build_spawn_plan(
        mode="deep",
        domain="finance",
        constraints=[
            "Need better pricing and margin controls",
            "Must satisfy security and compliance",
        ],
        memory_context=["Recent outage was caused by latency spikes."],
    )
    assert payload["selected_count"] >= 1
    first = payload["extra_critics"][0]
    assert "score" in first
    assert "matched_keywords" in first
    assert "evidence" in first
    assert "tool_contract" in first
    assert "tools" in first["tool_contract"]
    assert "routing_policy" in first["tool_contract"]


def test_spawn_plan_honors_role_score_threshold(monkeypatch) -> None:  # noqa: ANN001
    monkeypatch.setenv("RIM_SPAWN_MIN_ROLE_SCORE", "10")
    payload = build_spawn_plan(
        mode="deep",
        domain="general",
        constraints=["light requirement"],
        memory_context=[],
    )
    assert payload["selected_count"] == 0
