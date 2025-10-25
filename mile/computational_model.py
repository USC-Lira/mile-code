import gymnasium as gym
from typing import Any, Callable, List, Mapping, Optional, Sequence, Tuple, Union
import numpy as np

import torch
import torch.nn.functional as F
import torch.distributions as D

from stable_baselines3.sac.policies import SACPolicy
from stable_baselines3.dqn.policies import QNetwork
from stable_baselines3.common.policies import ActorCriticPolicy


COST_LOOKUP = {    
    'button-press-v2': [150, 200.0],
    'peg-insert-side-v2': [75, 175.0],
    'pick-place-v2': [250, 200.0],
    'drawer-open-v2': [60, 75.0],
    'LunarLander-v2': [3, 1.0],
}
LOG_STD_MAX = 2
LOG_STD_MIN = -20


def sum_independent_dims(tensor: torch.Tensor) -> torch.Tensor:
    ## Modified from stable-baselines3 ##
    """
    Continuous actions are usually considered to be independent,
    so we can sum components of the ``log_prob`` or the entropy.

    :param tensor: shape: (n_batch, n_actions) or (n_batch,)
    :return: shape: (n_batch,)
    """
    if len(tensor.shape) > 1:
        tensor = tensor.sum(dim=-1)
    else:
        tensor = tensor.sum()
    return tensor


def monte_carlo_samples(state: torch.Tensor, 
                        policy: Union[ActorCriticPolicy, SACPolicy], 
                        num_samples: int =1000) -> torch.Tensor:
    '''Compute the expectation of a function using Monte Carlo sampling
    Input:
        state: state of the environment (batch_size, state_size) or (state_size)
        policy: policy of the robot (preferably SAC policy)
        function: function to compute the expectation
        num_samples: number of samples to use for Monte Carlo sampling
    
    Output:
        samples: samples from the distribution (num_samples, batch_size, action_size) or (num_samples, action_size)
    '''
    if isinstance(policy, SACPolicy):
        mu, log_std, _ = policy.actor.get_action_dist_params(state)
        dist = D.Normal(mu, log_std.exp())
    elif isinstance(policy, ActorCriticPolicy):
        dist = policy.get_distribution(state)
        dist = dist.distribution
    samples = dist.rsample((num_samples,))
    return samples

def computational_intervention_model(state: torch.Tensor,
                                     mental_model: Union[ActorCriticPolicy, QNetwork],
                                     policy: Union[SACPolicy, ActorCriticPolicy, QNetwork],
                                     cost: int = 0,
                                     cdf_scale: float = 1.0) -> Tuple[torch.Tensor, torch.Tensor] | Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    '''Compute the intervention criteria for a given continuous policy.
    Input:
        state: state of the environment (batch_size, state_size) or (state_size)
        policy: policy used for intervention
        mental_model: mental model of the robot (What the human thinks the robot will do)
        cost: cost of intervention (0 by default)

    Output:
        total_prob: final probability distribution in the case of intervention (distribution)
        intervention_prob: probability of intervention (batch_size, 2) or (2)
    '''
    if len(state.shape) == 1:
        state = state.unsqueeze(0)

    # Discrete environment
    if isinstance(policy.action_space, gym.spaces.Discrete):
        assert isinstance(mental_model, QNetwork), "Mental model should be Discrete"
        assert isinstance(policy, QNetwork), "Policy should be Discrete"
        mental_model_expected_action_probs = mental_model(state)
        if len(mental_model_expected_action_probs.shape) == 1:
            mental_model_expected_action_probs = mental_model_expected_action_probs.unsqueeze(0)
        mental_model_expected_action_probs = F.softmax(mental_model_expected_action_probs, dim=1)
        policy_expected_action_probs = policy(state)
        if len(policy_expected_action_probs.shape) == 1:
            policy_expected_action_probs = policy_expected_action_probs.unsqueeze(0)
        policy_expected_action_probs = F.softmax(policy_expected_action_probs, dim=1)
        mean = torch.tensor([0.0], device=state.device)
        std_dev = torch.tensor([1.0], device=state.device)
        inside_cdf = (mental_model_expected_action_probs * torch.log(policy_expected_action_probs)).sum(dim=1, keepdim=True)
        inside_cdf = torch.log(policy_expected_action_probs) - inside_cdf
        inside_cdf = inside_cdf - cost
        intervention_prob = (policy_expected_action_probs * torch.distributions.Normal(mean, std_dev).cdf(inside_cdf)).sum(dim=1)
        total_prob = torch.ones((policy_expected_action_probs.shape[0],policy_expected_action_probs.shape[1]+1), device=state.device)
        total_prob[:,:-1] = policy_expected_action_probs*intervention_prob.unsqueeze(1)
        total_prob[:,-1] = 1 - torch.sum(total_prob[:,:-1], dim=1)
    
        return total_prob, intervention_prob
    
    # Continuous environment
    elif isinstance(policy.action_space, gym.spaces.Box):
        mental_model_samples = monte_carlo_samples(state=state, policy=mental_model, num_samples=1000)
        if isinstance(policy, SACPolicy):
            mu, log_std, _ = policy.actor.get_action_dist_params(state)
            policy_dist = D.Normal(mu, log_std.exp())
        elif isinstance(policy, ActorCriticPolicy):
            dist = policy.get_distribution(state)
            policy_dist = dist.distribution
        else:
            raise ValueError("Policy should be either SACPolicy or ActorCriticPolicy")
        mental_model_expectation = torch.mean(sum_independent_dims(policy_dist.log_prob(mental_model_samples)), dim=0)
        policy_samples = monte_carlo_samples(state=state, policy=policy, num_samples=1000)
        mean = torch.tensor([0.0], device=state.device)
        std_dev = torch.tensor([cdf_scale], device=state.device)
        policy_expectation = D.Normal(mean, std_dev).cdf(sum_independent_dims(policy_dist.log_prob(policy_samples))-mental_model_expectation-cost)
        intervention_prob = torch.mean(policy_expectation, dim=0)
        final_mu = intervention_prob.unsqueeze(1)*mu
        final_log_std = torch.log(intervention_prob.unsqueeze(1)) + log_std
        final_log_std = torch.clamp(final_log_std, LOG_STD_MIN, LOG_STD_MAX)

        # turn intervention_prob into a vector of probabilities
        intervention_prob = intervention_prob.unsqueeze(-1)
        intervention_prob = torch.cat((1-intervention_prob, intervention_prob), dim=-1)

        return final_mu, final_log_std, intervention_prob, policy_dist.mean, torch.log(policy_dist.stddev)
    

def random_intervention(intervention_rate=0.1):
    intervention = np.random.choice([0, 1], p=[1-intervention_rate, intervention_rate])
    intervention = bool(intervention)
    return intervention
