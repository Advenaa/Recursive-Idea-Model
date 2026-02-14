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
