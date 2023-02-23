from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Optional

from dataclass_wizard import JSONWizard


@dataclass
class ArgsType(JSONWizard):
    """
    Data dataclass

    """

    runner: str
    mac: str
    env: str
    batch_size_run: int
    test_nepisode: int
    test_interval: int
    test_greedy: bool
    log_interval: int
    runner_log_interval: int
    learner_log_interval: int
    t_max: int
    use_cuda: bool
    buffer_cpu_only: bool
    use_tensorboard: bool
    save_model: bool
    save_model_interval: int
    checkpoint_path: str
    evaluate: bool
    load_step: int
    save_replay: bool
    local_results_path: str
    gamma: float
    batch_size: int
    buffer_size: int
    lr: float
    critic_lr: float
    optim_alpha: float
    optim_eps: float
    grad_norm_clip: int
    agent: str
    rnn_hidden_dim: int
    obs_agent_id: bool
    obs_last_action: bool
    repeat_id: int
    label: str
    env_args: EnvArgs
    action_selector: str
    epsilon_start: float
    epsilon_finish: float
    epsilon_anneal_time: int
    target_update_interval: int
    agent_output_type: str
    learner: str
    double_q: bool
    mixer: str
    mixing_embed_dim: int
    hypernet_layers: int
    hypernet_embed: int
    name: str

    # Additional
    device: Optional[str] = None
    unique_token: Optional[str] = None
    n_agents: Optional[int] = None
    n_actions: Optional[int] = None
    state_shape: Optional[int] = None


# @dataclass
# class EnvInfoType:
#     state_shape: int
#     obs_shape: int
#     n_actions: int
#     n_agents: int
#     episode_limit: int
#     agent_features: List[str]
#     enemy_features: List[str]


@dataclass
class EnvArgs:
    """
    EnvArgs dataclass

    """

    continuing_episode: bool
    difficulty: int | str
    game_version: Optional[int | str]
    map_name: str
    move_amount: int
    obs_all_health: bool
    obs_instead_of_state: bool
    obs_last_action: bool
    obs_own_health: bool
    obs_pathing_grid: bool
    obs_terrain_height: bool
    obs_timestep_number: bool
    reward_death_value: int
    reward_defeat: int
    reward_negative_scale: float
    reward_only_positive: bool
    reward_scale: bool
    reward_scale_rate: int
    reward_sparse: bool
    reward_win: int
    replay_dir: str
    replay_prefix: str
    state_last_action: bool
    state_timestep_number: bool
    step_mul: int
    seed: Optional[int]
    heuristic_ai: bool
    heuristic_rest: bool
    debug: bool
