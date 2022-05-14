import torch
import torch.nn as nn
from Distribution import Dirichlet

class Encoder(nn.Module):
    def __init__(self, state1_dim=5, state2_dim=2):
        super().__init__()
        self.state1_dim = state1_dim
        self.state2_dim = state2_dim
        self.layer1 = nn.Linear(state1_dim+state2_dim, 128)
        self.layer2 = nn.Linear(128, 64)
        self.hidden_act = nn.SELU()

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        return x

class Decoder(nn.Module):
    def __init__(self, output_dim=1):
        super().__init__()
        self.output_dim = output_dim
        self.layer1 = nn.Linear(64, 128)
        self.layer2 = nn.Linear(128, 256)
        self.layer3 = nn.Linear(256, output_dim)
        self.hidden_act = nn.SELU()

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x

class Score(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, s1, s2):
        x = torch.cat([s1, s2], dim=-1)
        x = self.encoder(x)
        x = self.decoder(x)
        return x

# class Score(nn.Module):
#     def __init__(self, state1_dim=5, state2_dim=2, output_dim=1):
#         super().__init__()
#
#         self.state1_dim = state1_dim
#         self.state2_dim = state2_dim
#         self.output_dim = output_dim

#         self.layer1 = nn.Linear(state1_dim+state2_dim, 64)
#         self.layer2 = nn.Linear(64, 32)
#         self.layer3 = nn.Linear(32, output_dim)
#         self.hidden_act = nn.ReLU()
#         self.out_act = nn.Identity()
#
#         nn.init.kaiming_normal_(self.layer1.weight, nonlinearity="relu")
#         nn.init.kaiming_normal_(self.layer2.weight, nonlinearity="relu")
#         nn.init.kaiming_normal_(self.layer3.weight, nonlinearity="relu")
#
#     def forward(self, s1, s2):
#         # x = self.encoder(s1)
#         x = torch.concat([s1, s2], dim=-1)
#         x = self.layer1(x)
#         x = self.hidden_act(x)
#         x = self.layer2(x)
#         x = self.hidden_act(x)
#         x = self.layer3(x)
#         x = self.out_act(x)
#         return x

class Actor(nn.Module):
    def __init__(self, score_net):
        super().__init__()
        self.score_net = score_net

    def forward(self, s1_tensor, portfolio):
        """
        state = (s1_tensor, portfolio)
        s1_tensor: (batch, assets, features)
        """
        # x1 = s1_tensor[:,0,:]
        # x2 = s1_tensor[:,1,:]
        # x3 = s1_tensor[:,2,:]

        # score1 = self.score_net(x1, torch.cat([portfolio[:,0], portfolio[:,1]], dim=-1))
        # score2 = self.score_net(x2, torch.cat([portfolio[:,0], portfolio[:,2]], dim=-1))
        # score3 = self.score_net(x3, torch.cat([portfolio[:,0], portfolio[:,3]], dim=-1))

        # alpha = torch.cat([score1, score2, score3, score4, score5, score6, score7], dim=-1)
        for i in range(s1_tensor.shape[1]):
            globals()[f"x{i+1}"] = s1_tensor[:,i,:]

        for k in range(s1_tensor.shape[1]):
            state2 = torch.cat([portfolio[:,0], portfolio[:,k+1]], dim=-1)
            globals()[f"score{k+1}"] = self.score_net(globals()[f"x{k+1}"], state2)

        for j in range(s1_tensor.shape[1]):
            scores = list() if j == 0 else scores
            scores.append(globals()[f"score{j+1}"])

        alpha = torch.cat(scores, dim=-1)
        alpha = torch.tanh(alpha) + 2.0
        return alpha

    def sampling(self, s1_tensor, portfolio, repre=False):
        batch_num = s1_tensor.shape[0]
        cash_alpha = torch.ones(size=(batch_num, 1)) * 2.0
        alpha = torch.cat([cash_alpha, self(s1_tensor, portfolio)], dim=-1)
        dirichlet = Dirichlet(alpha)

        #Dirichlet 분포 mode 계산
        B = alpha.shape[0] #Batch num
        N = alpha.shape[1] #Asset num
        total = torch.sum(alpha, dim=1).view(B, 1)
        vector_1 = torch.ones(size=alpha.shape)
        vector_N = torch.ones(size=(B, 1)) * N

        #Representative value
        mode = (alpha - vector_1) / (total - vector_N)
        mean = dirichlet.mean

        if repre == "mean":
            sampled_p = mean
        elif repre == "mode":
            sampled_p = mode
        elif repre is False:
            sampled_p = dirichlet.sample([1])[0]

        log_pi = dirichlet.log_prob(sampled_p)
        return sampled_p, log_pi


class Critic(nn.Module):
    def __init__(self, score_net, header_dim=None):
        super().__init__()
        self.score_net = score_net
        self.header = header(input_dim=header_dim)

    def forward(self, s1_tensor, portfolio):
        for i in range(s1_tensor.shape[1]):
            globals()[f"x{i+1}"] = s1_tensor[:,i,:]

        for k in range(s1_tensor.shape[1]):
            state2 = torch.cat([portfolio[:,0], portfolio[:,k+1]], dim=-1)
            globals()[f"score{k+1}"] = self.score_net(globals()[f"x{k+1}"], state2)

        for j in range(s1_tensor.shape[1]):
            scores = list() if j == 0 else scores
            scores.append(globals()[f"score{j+1}"])

        scores = torch.cat(scores, dim=-1)
        v = self.header(scores)
        return v

class header(nn.Module):
    def __init__(self, output_dim=1, input_dim=None):
        super().__init__()
        self.output_dim = output_dim
        self.input_dim = input_dim

        self.layer1 = nn.Linear(input_dim, 128)
        self.layer2 = nn.Linear(128 ,64)
        self.layer3 = nn.Linear(64, output_dim)
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
