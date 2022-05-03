import torch
import torch.nn as nn

from Distribution import Dirichlet
from AutoEncoder import MLPEncoder

# class Actor(nn.Module):
#     def __init__(self, state1_dim=5, state2_dim=2, output_dim=3):
#         super(Actor, self).__init__()
#         self.state1_dim = state1_dim
#         self.state2_dim = state2_dim
#         self.output_dim = output_dim
#         self.input_dim = 3 * (state1_dim + state2_dim)
#
#         self.layer1 = nn.Linear(self.input_dim, 128)
#         self.layer2 = nn.Linear(128, 64)
#         self.layer3 = nn.Linear(64, self.output_dim)
#         self.hidden_act = nn.ReLU()
#         self.out_act = nn.Identity()
#
#     def forward(self, s1_tensor, portfolio):
#         x1 = torch.cat([s1_tensor[:,0,:], torch.cat([portfolio[:,0], portfolio[:,1]], dim=-1)], dim=-1)
#         x2 = torch.cat([s1_tensor[:,1,:], torch.cat([portfolio[:,0], portfolio[:,2]], dim=-1)], dim=-1)
#         x3 = torch.cat([s1_tensor[:,2,:], torch.cat([portfolio[:,0], portfolio[:,3]], dim=-1)], dim=-1)
#
#         x = torch.cat([x1, x2, x3], dim=1)
#         x = self.layer1(x)
#         x = self.hidden_act(x)
#         x = self.layer2(x)
#         x = self.hidden_act(x)
#         x = self.layer3(x)
#         x = self.out_act(x)
#         alpha = torch.exp(x) + 1
#         return alpha
#
#     def sampling(self, s1_tensor, portfolio, Test=False):
#         batch_num = s1_tensor.shape[0]
#         cash_alpha = torch.ones(size=(batch_num, 1))
#         alpha = torch.cat([cash_alpha, self(s1_tensor, portfolio)], dim=-1) * 1.5
#         dirichlet = Dirichlet(alpha)
#
#         # Dirichlet 분포 최빈값 계산
#         B = alpha.shape[0] #Batch num
#         N = alpha.shape[1] #Asset num
#         total = torch.sum(alpha, dim=1).view(B, 1)
#         vector_1 = torch.ones(size=alpha.shape)
#         vector_N = torch.ones(size=(B, 1)) * N
#         mode = (alpha - vector_1) / (total - vector_N)
#
#         sample = mode if Test else dirichlet.sample([1])[0]
#         log_pi = dirichlet.log_prob(sample)
#         return sample, log_pi
#
#
# class Critic(nn.Module):
#     def __init__(self, state1_dim=5, state2_dim=2, output_dim=1):
#         super(Critic, self).__init__()
#         self.state1_dim = state1_dim
#         self.state2_dim = state2_dim
#         self.output_dim = output_dim
#         self.input_dim = 3 * (state1_dim + state2_dim)
#
#         self.layer1 = nn.Linear(self.input_dim, 128)
#         self.layer2 = nn.Linear(128, 64)
#         self.layer3 = nn.Linear(64, self.output_dim)
#         self.hidden_act = nn.ReLU()
#         self.out_act = nn.Identity()
#
#     def forward(self, s1_tensor, portfolio):
#         x1 = torch.cat([s1_tensor[:,0,:], torch.cat([portfolio[:,0], portfolio[:,1]], dim=-1)], dim=-1)
#         x2 = torch.cat([s1_tensor[:,1,:], torch.cat([portfolio[:,0], portfolio[:,2]], dim=-1)], dim=-1)
#         x3 = torch.cat([s1_tensor[:,2,:], torch.cat([portfolio[:,0], portfolio[:,3]], dim=-1)], dim=-1)
#
#         x = torch.cat([x1, x2, x3], dim=1)
#         x = self.layer1(x)
#         x = self.hidden_act(x)
#         x = self.layer2(x)
#         x = self.hidden_act(x)
#         x = self.layer3(x)
#         x = self.out_act(x)
#         value = x
#         return value

class Score(nn.Module):
    def __init__(self, state1_dim=5, state2_dim=2, output_dim=1):
        super().__init__()

        self.state1_dim = state1_dim
        self.state2_dim = state2_dim
        self.output_dim = output_dim
        self.encoder = MLPEncoder()
        self.encoder.eval()

        # self.layer1 = nn.Linear(128+state2_dim, 64)
        self.layer1 = nn.Linear(state1_dim+state2_dim, 64)
        self.layer2 = nn.Linear(64, 32)
        self.layer3 = nn.Linear(32, output_dim)
        self.hidden_act = nn.ReLU()
        self.out_act = nn.Identity()

        nn.init.kaiming_normal_(self.layer1.weight, nonlinearity="relu")
        nn.init.kaiming_normal_(self.layer2.weight, nonlinearity="relu")
        nn.init.kaiming_normal_(self.layer3.weight, nonlinearity="relu")

    def forward(self, s1, s2):
        # x = self.encoder(s1)
        x = torch.concat([s1, s2], dim=-1)
        x = self.layer1(x)
        x = self.hidden_act(x)
        x = self.layer2(x)
        x = self.hidden_act(x)
        x = self.layer3(x)
        x = self.out_act(x)
        return x

class Actor(nn.Module):
    def __init__(self, score_net:nn.Module, state1_dim=5, state2_dim=2, output_dim=1):
        super().__init__()
        self.score_net = score_net
        self.state1_dim = state1_dim
        self.state2_dim = state2_dim
        self.output_dim = output_dim

    def forward(self, s1_tensor, portfolio):
        """
        state = (s1_tensor, portfolio)
        s1_tensor: (batch, assets, features)
        """
        x1 = s1_tensor[:,0,:]
        x2 = s1_tensor[:,1,:]
        x3 = s1_tensor[:,2,:]

        score1 = self.score_net(x1, torch.cat([portfolio[:,0], portfolio[:,1]], dim=-1))
        score2 = self.score_net(x2, torch.cat([portfolio[:,0], portfolio[:,2]], dim=-1))
        score3 = self.score_net(x3, torch.cat([portfolio[:,0], portfolio[:,3]], dim=-1))

        alpha = torch.cat([score1, score2, score3], dim=-1)
        alpha = torch.exp(alpha) + 1
        return alpha

    def sampling(self, s1_tensor, portfolio, Test=False):
        batch_num = s1_tensor.shape[0]
        cash_alpha = torch.ones(size=(batch_num, 1)) * 2
        alpha = torch.cat([cash_alpha, self(s1_tensor, portfolio)], dim=-1)
        dirichlet = Dirichlet(alpha)

        #Dirichlet 분포 최빈값 계산
        B = alpha.shape[0] #Batch num
        N = alpha.shape[1] #Asset num
        total = torch.sum(alpha, dim=1).view(B, 1)
        vector_1 = torch.ones(size=alpha.shape)
        vector_N = torch.ones(size=(B, 1)) * N
        mode = (alpha - vector_1) / (total - vector_N)

        sampled_p = mode if Test else dirichlet.sample([1])[0]
        log_pi = dirichlet.log_prob(sampled_p)
        return sampled_p, log_pi


class Critic(nn.Module):
    def __init__(self, score_net:nn.Module, state1_dim=5, state2_dim=2, output_dim=1):
        super().__init__()
        self.score_net = score_net
        self.header = header()
        self.state1_dim = state1_dim
        self.state2_dim = state2_dim
        self.output_dim = output_dim

    def forward(self, s1_tensor, portfolio):
        x1 = s1_tensor[:,0,:]
        x2 = s1_tensor[:,1,:]
        x3 = s1_tensor[:,2,:]

        score1 = self.score_net(x1, torch.cat([portfolio[:,0], portfolio[:,1]], dim=-1))
        score2 = self.score_net(x2, torch.cat([portfolio[:,0], portfolio[:,2]], dim=-1))
        score3 = self.score_net(x3, torch.cat([portfolio[:,0], portfolio[:,3]], dim=-1))

        scores = torch.cat([score1, score2, score3], dim=-1)
        v = self.header(scores)
        return v

class header(nn.Module):
    def __init__(self, out_dim=1):
        super().__init__()
        self.out_dim = out_dim

        self.layer1 = nn.Linear(3, 128)
        self.layer2 = nn.Linear(128 ,64)
        self.layer3 = nn.Linear(64, out_dim)
        self.hidden_act = nn.ReLU()
        self.out_act = nn.Identity()

    def forward(self, scores):
        x = self.layer1(scores)
        x = self.hidden_act(x)
        x = self.layer2(x)
        x = self.hidden_act(x)
        x = self.layer3(x)
        x = self.out_act(x)
        return x

# if __name__ == "__main__":
#     s1_tensor = torch.rand(size=(10, 3, 5))
#     portfolio = torch.rand(size=(10, 4, 1))
#
#     score_net = Score()
#     actor = Actor(score_net)
#     critic = Critic(score_net)



