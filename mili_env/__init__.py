from gymnasium.envs.registration import register

register(
    id="mili_env/TerrainWorld-v0",
    entry_point="mili_env.envs:TerrainWorldEnv",
)
register(
    id="mili_env/GridWorld-v0",
    entry_point="mili_env.envs:GridWorldEnv",
)
