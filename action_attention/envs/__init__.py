from gym.envs.registration import register

register(
    'ShapesTrain-v0',
    entry_point='action_attention.envs.block_pushing:BlockPushing',
    max_episode_steps=100,
    kwargs={'render_type': 'shapes'},
)

register(
    'ShapesEval-v0',
    entry_point='action_attention.envs.block_pushing:BlockPushing',
    max_episode_steps=10,
    kwargs={'render_type': 'shapes'},
)

register(
    'ShapesOneStaticTrain-v0',
    entry_point='action_attention.envs.block_pushing:BlockPushing',
    max_episode_steps=100,
    kwargs={'render_type': 'shapes', 'num_static_objects': 1},
)

register(
    'ShapesOneStaticEval-v0',
    entry_point='action_attention.envs.block_pushing:BlockPushing',
    max_episode_steps=10,
    kwargs={'render_type': 'shapes', 'num_static_objects': 1},
)

register(
    'CubesTrain-v0',
    entry_point='action_attention.envs.block_pushing:BlockPushing',
    max_episode_steps=100,
    kwargs={'render_type': 'cubes'},
)

register(
    'CubesEval-v0',
    entry_point='action_attention.envs.block_pushing:BlockPushing',
    max_episode_steps=10,
    kwargs={'render_type': 'cubes'},
)
