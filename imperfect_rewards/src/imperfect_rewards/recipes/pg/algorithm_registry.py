from imperfect_rewards.recipes.pg.custom_grpo_trainer import CustomGRPOTrainer
from imperfect_rewards.recipes.pg.custom_ppo_trainer import CustomPPOTrainer
from imperfect_rewards.recipes.pg.custom_rloo_trainer import CustomRLOOTrainer

ALGORITHM_TRAINER_CLASSES = {
    'RLOO': CustomRLOOTrainer,
    'PPO': CustomPPOTrainer,
    'GRPO': CustomGRPOTrainer,
}
