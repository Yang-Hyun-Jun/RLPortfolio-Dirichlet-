import torch
import Visualizer
import numpy as np

from Environment import environment
from Agent import agent
from ReplayMemory import ReplayMemory
from Network import Actor
from Network import Critic
from Network import Score
from Metrics import Metrics

class learner:

    K = 3
    encoder_path = "/Users/mac/Desktop/RLPortfolio/AutoEncoder/encoder.pth"

    def __init__(self,
                 actor_lr=1e-4, critic_lr=1e-4,
                 tau = 0.005, delta=0.07,
                 discount_factor=0.9,
                 batch_size=256, memory_size=100000,
                 chart_data=None,
                 min_trading_price=None, max_trading_price=None):

        assert min_trading_price >= 0
        assert max_trading_price > 0
        assert max_trading_price >= min_trading_price

        self.environment = environment(chart_data)
        self.memory = ReplayMemory(max_size=memory_size)
        self.chart_data = chart_data
        self.batch_size = batch_size

        self.score_net = Score()
        self.actor = Actor(score_net=self.score_net)
        self.critic = Critic(score_net=self.score_net)
        self.critic_target = Critic(score_net=self.score_net)

        self.critic_target.load_state_dict(self.critic.state_dict())
        # self.actor.score_net.encoder.load_state_dict(torch.load(learner.encoder_path))
        # self.critic.score_net.encoder.load_state_dict(torch.load(learner.encoder_path))

        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.tau = tau
        self.delta = delta
        self.discount_factor = discount_factor
        self.min_trading_price = min_trading_price
        self.max_trading_price = max_trading_price

        self.agent = agent(environment=self.environment,
                           critic=self.critic,
                           critic_target=self.critic_target,
                           actor=self.actor,
                           critic_lr=self.critic_lr,
                           actor_lr=self.actor_lr,
                           tau=self.tau, delta=self.delta,
                           discount_factor=self.discount_factor,
                           min_trading_price=min_trading_price,
                           max_trading_price=max_trading_price)

    def reset(self):
        self.environment.reset()
        self.agent.reset()

    @staticmethod
    def prepare_training_inputs(sampled_exps):
        states1 = []
        states2 = []
        actions = []
        rewards = []
        next_states1 = []
        next_states2 = []
        log_probs = []
        dones = []

        for sampled_exp in sampled_exps:
            states1.append(sampled_exp[0])
            states2.append(sampled_exp[1])
            actions.append(sampled_exp[2])
            rewards.append(sampled_exp[3])
            next_states1.append(sampled_exp[4])
            next_states2.append(sampled_exp[5])
            log_probs.append(sampled_exp[6])
            dones.append(sampled_exp[7])

        states1 = torch.cat(states1, dim=0).float()
        states2 = torch.cat(states2, dim=0).float()
        actions = torch.cat(actions, dim=0).float()
        rewards = torch.cat(rewards, dim=0).float()
        next_states1 = torch.cat(next_states1, dim=0).float()
        next_states2 = torch.cat(next_states2, dim=0).float()
        log_probs = torch.cat(log_probs, dim=0).float()
        dones = torch.cat(dones, dim=0).float()
        return states1, states2, actions, rewards, next_states1, next_states2, log_probs, dones


    def run(self, num_episode=None, balance=None):
        self.agent.set_balance(balance)
        metrics = Metrics()
        steps_done = 0

        for episode in range(num_episode):
            self.reset()
            cum_r = 0
            state1 = self.environment.observe()
            portfolio = self.agent.portfolio
            while True:
                action, confidence, log_prob = self.agent.get_action(torch.tensor(state1).float().view(1,3,-1),
                                                                     torch.tensor(portfolio).float().view(1,4,-1))

                next_state1, next_portfolio, reward, done = self.agent.step(action, confidence)
                steps_done += 1

                # if steps_done == 1:
                #     reward -= 0
                # else:
                #     reward -= 0.10 * np.linalg.norm(action-action_, ord=1)

                experience = (torch.tensor(state1).float().view(1,3,-1),
                              torch.tensor(portfolio).float().view(1,4,-1),
                              torch.tensor(action).float().view(1,-1),
                              torch.tensor(reward).float().view(1,-1),
                              torch.tensor(next_state1).float().view(1,3,-1),
                              torch.tensor(next_portfolio).float().view(1,4,-1),
                              torch.tensor(log_prob).float().view(1,-1),
                              torch.tensor(done).float().view(1,-1))

                self.memory.push(experience)
                cum_r += reward
                state1 = next_state1
                portfolio = next_portfolio
                action_ = action

                if done:
                    break

                if steps_done % 300 == 0:
                    value = self.agent.critic(torch.tensor(state1).float().view(1,3,-1),
                                              torch.tensor(portfolio).float().view(1,4,-1)).detach().numpy()[0]
                    alpha = self.agent.actor(torch.tensor(state1).float().view(1,3,-1),
                                             torch.tensor(portfolio).float().view(1,4,-1)).detach()[0]
                    a = action
                    al = torch.cat([torch.tensor([2.0]), alpha], dim=-1).numpy()
                    p = self.agent.portfolio
                    pv = self.agent.portfolio_value
                    sv = self.agent.portfolio_value_static
                    balance = self.agent.balance
                    change = self.agent.change
                    pi_vector = self.agent.pi_operator(change)
                    profitloss = self.agent.profitloss
                    np.set_printoptions(precision=4, suppress=True)
                    print(f"episode:{episode} ------------------------------------------------------------------------")
                    print(f"price:{self.environment.get_price()}")
                    print(f"value:{value}")
                    print(f"action:{a}")
                    print(f"alpha:{al}")
                    print(f"portfolio:{p}")
                    print(f"pi_vector:{pi_vector}")
                    print(f"portfolio value:{pv}")
                    print(f"static value:{sv}")
                    print(f"balance:{balance}")
                    print(f"cum reward:{cum_r}")
                    print(f"profitloss:{profitloss}")
                    print("-------------------------------------------------------------------------------------------")

                # 학습
                if len(self.memory) >= self.batch_size:
                    sampled_exps = self.memory.sample(self.batch_size)
                    sampled_exps = self.prepare_training_inputs(sampled_exps)
                    self.agent.update(*sampled_exps)
                    self.agent.soft_target_update(self.agent.critic.parameters(), self.agent.critic_target.parameters())

                #metrics 마지막 episode에 대해서만
                if episode == range(num_episode)[-1]:
                    metrics.portfolio_values.append(self.agent.portfolio_value)
                    metrics.profitlosses.append(self.agent.profitloss)

            #시각화 마지막 episode에 대해서만
            if episode == range(num_episode)[-1]:
                #metric 계산과 저장
                metrics.get_profitlosses()
                metrics.get_portfolio_values()

                #계산한 metric 시각화와 저장
                Visualizer.get_portfolio_value_curve(metrics.portfolio_values)
                Visualizer.get_profitloss_curve(metrics.profitlosses)

    def save_model(self, critic_path, actor_path):
        torch.save(self.agent.critic.state_dict(), critic_path)
        torch.save(self.agent.actor.state_dict(), actor_path)

# if __name__ == "__main__":
#     import DataManager
#     path1 = "/Users/mac/Desktop/OHLCV_data/ALL_OHLCV/010140" #삼성중공업
#     path2 = "/Users/mac/Desktop/OHLCV_data/ALL_OHLCV/013570" #디와이
#     path3 = "/Users/mac/Desktop/OHLCV_data/ALL_OHLCV/010690" #화신
#     path4 = "/Users/mac/Desktop/OHLCV_data/ALL_OHLCV/000910" #유니온
#     path5 = "/Users/mac/Desktop/OHLCV_data/ALL_OHLCV/010060" #OCI
#     path6 = "/Users/mac/Desktop/OHLCV_data/ALL_OHLCV/034220" #LG디스플레이
#     path7 = "/Users/mac/Desktop/OHLCV_data/ALL_OHLCV/009540" #한국조선해양
#
#     path_list = [path1, path2, path3, path4, path5, path6, path7]
#     train_data, test_data = DataManager.get_data_tensor(path_list,
#                                                         train_date_start="20090101",
#                                                         train_date_end="20180101",
#                                                         test_date_start="20180102",
#                                                         test_date_end=None)

    # env = environment(train_data)
    # critic = Critic()
    # critic_target = Critic()
    # actor = Actor()
    # actor_target = Actor()
    # critic_target.load_state_dict(critic.state_dict())
    # actor_target.load_state_dict(actor.state_dict())
    # ou_noise = OUProcess(np.zeros(7))
    # memory = ReplayMemory(10000)
    # batch_size = 256
    # balance = 14000000
    #
    # agent = agent(environment=env,
    #               critic=critic,
    #               critic_target=critic_target,
    #               actor=actor,
    #               actor_target=actor_target,
    #               critic_lr=1e-4,
    #               actor_lr=1e-4,
    #               tau=0.005,
    #               discount_factor=0.9,
    #               min_trading_price=0,
    #               max_trading_price=balance/7)
    #
    # def prepare_training_inputs(sampled_exps):
    #     states1 = []
    #     states2 = []
    #     actions = []
    #     rewards = []
    #     next_states1 = []
    #     next_states2 = []
    #     dones = []
    #
    #     for sampled_exp in sampled_exps:
    #         states1.append(sampled_exp[0])
    #         states2.append(sampled_exp[1])
    #         actions.append(sampled_exp[2])
    #         rewards.append(sampled_exp[3])
    #         next_states1.append(sampled_exp[4])
    #         next_states2.append(sampled_exp[5])
    #         dones.append(sampled_exp[6])
    #
    #     states1 = torch.cat(states1, dim=0).float()
    #     states2 = torch.cat(states2, dim=0).float()
    #     actions = torch.cat(actions, dim=0).float()
    #     rewards = torch.cat(rewards, dim=0).float()
    #     next_states1 = torch.cat(next_states1, dim=0).float()
    #     next_states2 = torch.cat(next_states2, dim=0).float()
    #     dones = torch.cat(dones, dim=0).float()
    #     return states1, states2, actions, rewards, next_states1, next_states2, dones
    #
    # agent.set_balance(balance)
    # agent.epsilon = 0
    # steps_done = 0
    #
    # for episode in range(1):
    #     agent.reset()
    #     env.reset()
    #     cum_r = 0
    #     state1 = env.observe()
    #     portfolio = agent.portfolio
    #
    #     while True:
    #         action, confidence = agent.get_action(torch.tensor(state1).float().view(1,7,-1),
    #                                               torch.tensor(portfolio).float().view(1,8,-1))
    #         action += ou_noise()
    #         next_state1, next_portfolio, reward, done = agent.step(action, confidence)
    #         steps_done += 1
    #
    #         experience = (torch.tensor(state1).float().view(1,7,-1),
    #                       torch.tensor(portfolio).float().view(1,8,-1),
    #                       torch.tensor(action).float().view(1,-1),
    #                       torch.tensor(reward).float().view(1,-1),
    #                       torch.tensor(next_state1).float().view(1,7,-1),
    #                       torch.tensor(next_portfolio).float().view(1,8,-1),
    #                       torch.tensor(done).float().view(1,-1))
    #
    #         memory.push(experience)
    #         cum_r += reward
    #         state1 = next_state1
    #         portfolio = next_portfolio
    #
    #         if done:
    #             break
    #
    #     sampled_exps = memory.sample(10)
    #     sampled_exps = prepare_training_inputs(sampled_exps)
    #     agent.update(*sampled_exps)
    #     agent.soft_target_update(agent.critic.parameters(), agent.critic_target.parameters())
    #     agent.soft_target_update(agent.actor.parameters(), agent.actor_target.parameters())
        # print(sampled_exps[0].shape)
        # print(sampled_exps[1].shape)
        # print(sampled_exps[2].shape)
        # print(sampled_exps[3].shape)
        # print(sampled_exps[4].shape)
        # print(sampled_exps[5].shape)
        # print(sampled_exps[6].shape)


            # if len(memory) >= batch_size:
            #     sampled_exps = memory.sample(batch_size)
            #     sampled_exps = prepare_training_inputs(sampled_exps)
            #     agent.update(*sampled_exps)
            #     agent.soft_target_update(agent.critic.parameters(), agent.critic_target.parameters())
            #     agent.soft_target_update(agent.actor.parameters(), agent.actor_target.parameters())

            # metrics 마지막 episode 대해서만
            # if episode == range(num_episode)[-1]:
            #     metrics.portfolio_values.append(self.agent.portfolio_value)
            #     metrics.profitlosses.append(self.agent.profitloss)

            # if done:
            #     break
