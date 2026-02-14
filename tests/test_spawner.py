import json

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


def test_spawn_plan_can_generate_dynamic_specialist_roles(monkeypatch) -> None:  # noqa: ANN001
    monkeypatch.setenv("RIM_ENABLE_DYNAMIC_SPECIALISTS", "1")
    monkeypatch.setenv("RIM_SPAWN_MAX_DYNAMIC_SPECIALISTS", "2")
    payload = build_spawn_plan(
        mode="deep",
        domain="bioinformatics",
        constraints=[
            "Need genomics workflow controls and lineage tracking",
            "Evaluate proteomics risk model",
        ],
        memory_context=[],
    )
    roles = [item["role"] for item in payload["extra_critics"]]
    assert any(role.startswith("dynamic_") for role in roles)
    dynamic_item = next(item for item in payload["extra_critics"] if item["role"].startswith("dynamic_"))
    assert dynamic_item["tool_contract"]["routing_policy"] == "prioritize_domain_specific_signals"


def test_spawn_plan_applies_spawn_policy_file_defaults(tmp_path, monkeypatch) -> None:  # noqa: ANN001
    policy_path = tmp_path / "spawn_policy.json"
    policy_path.write_text(
        json.dumps(
            {
                "policy": {
                    "policy_env": {
                        "RIM_SPAWN_MIN_ROLE_SCORE": 0.5,
                        "RIM_SPAWN_MAX_SPECIALISTS_DEEP": 5,
                        "RIM_ENABLE_DYNAMIC_SPECIALISTS": 1,
                        "RIM_SPAWN_MAX_DYNAMIC_SPECIALISTS": 3,
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("RIM_SPAWN_POLICY_PATH", str(policy_path))
    monkeypatch.delenv("RIM_SPAWN_MIN_ROLE_SCORE", raising=False)
    monkeypatch.delenv("RIM_SPAWN_MAX_SPECIALISTS_DEEP", raising=False)
    monkeypatch.delenv("RIM_ENABLE_DYNAMIC_SPECIALISTS", raising=False)
    monkeypatch.delenv("RIM_SPAWN_MAX_DYNAMIC_SPECIALISTS", raising=False)

    payload = build_spawn_plan(
        mode="deep",
        domain="bioinformatics",
        constraints=[
            "Need genomics workflow controls and lineage tracking",
            "Evaluate proteomics risk model",
        ],
        memory_context=[],
    )
    assert payload["policy_applied"] is True
    assert payload["policy_path"] == str(policy_path)
    assert payload["min_role_score"] == 0.5
    assert payload["selected_count"] <= 5


def test_spawn_plan_applies_policy_role_and_tool_overrides(tmp_path, monkeypatch) -> None:  # noqa: ANN001
    policy_path = tmp_path / "spawn_policy.json"
    policy_path.write_text(
        json.dumps(
            {
                "policy": {
                    "policy_env": {
                        "RIM_SPAWN_MIN_ROLE_SCORE": 2.2,
                        "RIM_ENABLE_DYNAMIC_SPECIALISTS": 1,
                        "RIM_SPAWN_MAX_DYNAMIC_SPECIALISTS": 2,
                        "RIM_SPAWN_ROLE_BOOSTS": {"security": 2.5},
                        "RIM_SPAWN_DYNAMIC_TOKEN_BOOSTS": {"aodkinv": 3.2},
                        "RIM_SPAWN_ROLE_ROUTING_OVERRIDES": {
                            "security": "prioritize_compliance_violations",
                            "dynamic_aodkinv": "prioritize_domain_specific_signals",
                        },
                        "RIM_SPAWN_ROLE_TOOL_OVERRIDES": {
                            "security": ["compliance_matrix", "abuse_case_review"],
                            "dynamic_aodkinv": ["context_probe:aodkinv", "counterexample_search"],
                        },
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("RIM_SPAWN_POLICY_PATH", str(policy_path))
    monkeypatch.delenv("RIM_SPAWN_ROLE_BOOSTS", raising=False)
    monkeypatch.delenv("RIM_SPAWN_DYNAMIC_TOKEN_BOOSTS", raising=False)
    monkeypatch.delenv("RIM_SPAWN_ROLE_ROUTING_OVERRIDES", raising=False)
    monkeypatch.delenv("RIM_SPAWN_ROLE_TOOL_OVERRIDES", raising=False)

    payload = build_spawn_plan(
        mode="deep",
        domain="general",
        constraints=["Need aodkinv controls before launch"],
        memory_context=[],
    )
    roles = [item["role"] for item in payload["extra_critics"]]
    assert "security" in roles
    assert "dynamic_aodkinv" in roles
    security_item = next(item for item in payload["extra_critics"] if item["role"] == "security")
    assert security_item["tool_contract"]["routing_policy"] == "prioritize_compliance_violations"
    assert security_item["tool_contract"]["tools"] == ["compliance_matrix", "abuse_case_review"]
    dynamic_item = next(item for item in payload["extra_critics"] if item["role"] == "dynamic_aodkinv")
    assert dynamic_item["tool_contract"]["tools"] == ["context_probe:aodkinv", "counterexample_search"]
    assert payload["role_boosts"]["security"] == 2.5
    assert payload["dynamic_token_boosts"]["aodkinv"] == 3.2


def test_spawn_plan_env_json_maps_override_policy_maps(tmp_path, monkeypatch) -> None:  # noqa: ANN001
    policy_path = tmp_path / "spawn_policy.json"
    policy_path.write_text(
        json.dumps(
            {
                "policy": {
                    "policy_env": {
                        "RIM_SPAWN_ROLE_BOOSTS": {"security": 2.0},
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("RIM_SPAWN_POLICY_PATH", str(policy_path))
    monkeypatch.setenv("RIM_SPAWN_ROLE_BOOSTS", '{"finance":3.0}')

    payload = build_spawn_plan(
        mode="deep",
        domain="general",
        constraints=["light requirement"],
        memory_context=[],
    )
    roles = [item["role"] for item in payload["extra_critics"]]
    assert "finance" in roles
    assert payload["role_boosts"] == {"finance": 3.0}
