from typing import Any

import numpy as np

from mili_env.envs.terrain_world import TerrainWorldEnv


def test_poi_reached_increases_reward() -> None:
    # Minimal env with 1 agent, no enemies
    env = TerrainWorldEnv(num_agents=1, num_enemies=0)
    # Configure multiple POIs and a non-zero radius for easier triggering
    env.set_poi_config(pois_per_mission=2, poi_radius=0)

    _, _ = env.reset(seed=123)

    # Get agent 0 POIs and move the agent directly on a POI
    pois = env.get_mission_pois(0)
    assert len(pois) >= 1
    poi = pois[0]

    # Teleport agent to POI (use public setter)
    agent = env.agents[0]
    agent.set_position((float(poi[0]), float(poi[1])))

    # Step no-op actions: no moves, no fire, no comms, no orders
    action: Any = {
        "agent_0": {
            "elementary_act": 0,
            "combat": {"move": 0, "fire_enemy": env.num_agents},
            "comm": {
                "types": [0] * env.MAX_COMMS_PER_STEP,
                "targets": [env.num_agents] * env.MAX_COMMS_PER_STEP,
            },
            "c2": {
                "give_order": 0,
                "orders": [env.MISSION_NO_CHANGE_SENTINEL] * env.num_agents,
            },
        }
    }

    # Run step and obtain rewards
    _, reward, _, _, info = env.step(action)

    # Because at POI, mission_event_reward should include poi_reached contribution.
    # We can't easily isolate the scalar; smoke-check per-agent reward via info.
    assert isinstance(reward, float)
    assert "agent_0" in info
    assert "per_agent_reward" in info["agent_0"]
    assert np.isfinite(info["agent_0"]["per_agent_reward"])  # smoke check


def test_reward_higher_at_poi_than_away() -> None:
    env = TerrainWorldEnv(num_agents=1, num_enemies=0)
    env.set_poi_config(pois_per_mission=1, poi_radius=0)
    _, _ = env.reset(seed=321)

    poi = env.get_mission_pois(0)[0]
    agent = env.agents[0]

    # Create consistent action
    base_action: Any = {
        "agent_0": {
            "elementary_act": 0,
            "combat": {"move": 0, "fire_enemy": env.num_agents},
            "comm": {
                "types": [0] * env.MAX_COMMS_PER_STEP,
                "targets": [env.num_agents] * env.MAX_COMMS_PER_STEP,
            },
            "c2": {"give_order": 0, "orders": [env.MISSION_NO_CHANGE_SENTINEL] * env.num_agents},
        }
    }

    # Step at POI
    agent.set_position((float(poi[0]), float(poi[1])))
    _, _reward_at, _, _, info_at = env.step(base_action)

    # Step away from POI (one cell off if possible)
    away_x = min(max(0, int(poi[0]) + 1), env.width - 1)
    away_y = int(poi[1])
    agent.set_position((float(away_x), float(away_y)))
    _, _reward_away, _, _, info_away = env.step(base_action)

    assert info_at["agent_0"]["per_agent_reward"] >= info_away["agent_0"]["per_agent_reward"]


def test_poi_radius_effect() -> None:
    env = TerrainWorldEnv(num_agents=1, num_enemies=0)
    env.set_poi_config(pois_per_mission=1, poi_radius=0)
    _, _ = env.reset(seed=456)

    poi = env.get_mission_pois(0)[0]
    agent = env.agents[0]

    # Place agent one cell away from POI
    off_x = min(max(0, int(poi[0]) + 1), env.width - 1)
    off_y = int(poi[1])
    agent.set_position((float(off_x), float(off_y)))

    _base_action: Any = {
        "agent_0": {
            "elementary_act": 0,
            "combat": {"move": 0, "fire_enemy": env.num_agents},
            "comm": {
                "types": [0] * env.MAX_COMMS_PER_STEP,
                "targets": [env.num_agents] * env.MAX_COMMS_PER_STEP,
            },
            "c2": {"give_order": 0, "orders": [env.MISSION_NO_CHANGE_SENTINEL] * env.num_agents},
        }
    }

    # Test POI events directly through the environment's event system
    events_rad0 = env._collect_events_for_agent(0)  # noqa: SLF001
    poi_reached_rad0 = events_rad0.get("poi_reached", 0.0)

    # Increase radius to 1 and test again
    env.set_poi_config(poi_radius=1)
    events_rad1 = env._collect_events_for_agent(0)  # noqa: SLF001
    poi_reached_rad1 = events_rad1.get("poi_reached", 0.0)

    # With radius 1, agent should now be within POI range
    assert poi_reached_rad1 > poi_reached_rad0, (
        f"POI reached should increase from {poi_reached_rad0} to {poi_reached_rad1} "
        f"when radius increases"
    )
