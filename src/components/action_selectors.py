"""select action"""
import copy
import numpy as np
from .epsilon_schedules import DecayThenFlatSchedule


REGISTRY = {}


class MultinomialActionSelector():

    def __init__(self, args):
        self.args = args

        self.schedule = DecayThenFlatSchedule(args.epsilon_start, args.epsilon_finish, args.epsilon_anneal_time,
                                              decay="linear")
        self.epsilon = self.schedule.eval(0)
        self.test_greedy = getattr(args, "test_greedy", True)

    def select_action(self, agent_inputs, avail_actions, t_env, test_mode=False):
        masked_policies = agent_inputs.clone()
        masked_policies[avail_actions == 0.0] = 0.0

        self.epsilon = self.schedule.eval(t_env)

        if test_mode and self.test_greedy:
            picked_actions = masked_policies.max(dim=2)[1]
        else:
            picked_actions = Categorical(masked_policies).sample().long()

        return picked_actions


REGISTRY["multinomial"] = MultinomialActionSelector


class EpsilonGreedyActionSelector():

    def __init__(self, args):
        self.args = args

        self.schedule = DecayThenFlatSchedule(args.epsilon_start, args.epsilon_finish, args.epsilon_anneal_time,
                                              decay="linear")
        self.epsilon = self.schedule.eval(0)

    def select_action(self, agent_inputs, avail_actions, t_env, test_mode=False):
        # Assuming agent_inputs is a batch of Q-Values for each agent bav
        self.epsilon = self.schedule.eval(t_env)

        if test_mode:
            # Greedy action selection only
            self.epsilon = 0.0
        # mask actions that are excluded from selection
        masked_q_values = copy.deepcopy(agent_inputs)
        masked_q_values[avail_actions == 0.0] = -float("inf")  # should never be selected!
        shape = agent_inputs[:, :, 0].shape
        random_numbers = np.random.uniform(0, 1, shape)
        pick_random = (random_numbers < self.epsilon)
        random_actions = []
        for i in range(avail_actions.shape[1]):
            pro = avail_actions[0, i, :] / avail_actions[0, i, :].sum()
            random_actions.append(np.random.choice(range(avail_actions.shape[2]), 1, p=pro)[0])
        random_actions = np.expand_dims(random_actions, axis=0)
        picked_actions = pick_random * random_actions + (1 - pick_random) * masked_q_values.argmax(axis=2)
        return picked_actions


REGISTRY["epsilon_greedy"] = EpsilonGreedyActionSelector
