from mili_env.envs.terrain_world import TerrainWorldEnv


def test_register_and_select_reward_functions(tmp_path):
    env = TerrainWorldEnv(render_mode=None, num_agents=3)

    # Simple per-agent reward: give +1 to agents at target (based on prev_info)
    def per_agent(prev_info):
        rewards = {}
        for k, v in prev_info.items():
            # v is expected to contain 'distance'
            rewards[k] = 1.0 if float(v.get("distance", 0.0)) == 0.0 else 0.0
        return rewards

    # Simple team reward: sum of per-agent rewards
    def team(_prev_info, per_agent_rewards):
        return float(sum(float(x) for x in per_agent_rewards.values()))

    env.register_reward_function("test_simple", per_agent_fn=per_agent, team_fn=team)
    assert "test_simple" in env.get_registered_reward_functions()

    # Set temporary log path and select function
    log_file = tmp_path / "reward_runs.jsonl"
    env.reward_log_path = str(log_file)
    env.set_reward_function("test_simple")

    # Run reset to archive metadata
    obs, _info = env.reset(seed=123)
    assert log_file.exists()

    # Take one step with default actions (build from observation keys)
    actions = dict.fromkeys(obs.keys(), 0)
    _obs2, scalar_reward, _terminated, _truncated, info2 = env.step(actions)

    # Because per_agent gives 0/1 and team sums them, scalar_reward should be a float
    assert isinstance(scalar_reward, float)

    # The per-agent rewards are embedded inside each agent's info entry
    for agent_key in obs:
        assert agent_key in info2
        assert "per_agent_reward" in info2[agent_key]

    # Now switch to per_agent mode and check scalar reward is agent_0's reward
    env.set_reward_mode("per_agent")
    _obs3, scalar_reward2, *_ = env.step(actions)
    assert isinstance(scalar_reward2, float)

    # Clean-up
    env.close()
