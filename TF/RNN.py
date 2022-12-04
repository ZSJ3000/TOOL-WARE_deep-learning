import torch
from torch import nn


class Rnn(nn.Module):
    def __init__(self, INPUT_SIZE):
        super(Rnn, self).__init__()
        self.hidden_size = 64
        self.num_layers = 2
        self.rnn = nn.RNN(
            input_size=INPUT_SIZE,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
            nonlinearity='relu',
            bias=True,
            dropout=0.3,

            bidirectional=True
        )
        self.rnn1 = nn.RNN(
            input_size=128,
            hidden_size=32,
            num_layers=2,
            batch_first=True,
            nonlinearity='relu',
            dropout=0.1,



        )
        self.rnn2 = nn.RNN(
            input_size=64,
            hidden_size=32,
            num_layers=2,
            batch_first=True,
            nonlinearity='relu',
            bidirectional=True,
            dropout=0.3,




        )
        self.rnn3 = nn.RNN(
            input_size=64,
            hidden_size=16,
            num_layers=2,
            batch_first=True,
            nonlinearity='relu',
            dropout=0.1,
            bidirectional=True,


        )
        self.rnn4 = nn.RNN(
            input_size=256,
            hidden_size=64,
            num_layers=2,
            batch_first=True,
            nonlinearity='relu',
            bidirectional=True,

            dropout=0.3

        )
        self.rnn5 = nn.RNN(
            input_size=128,
            hidden_size=32,
            num_layers=2,
            batch_first=True,
            nonlinearity='relu',
            dropout=0.1,
            bidirectional=True,

        )



        self.out =  nn.Sequential(
            nn.Linear(32, 16),
            nn.BatchNorm1d(1),
            nn.Sigmoid(),
            nn.Linear(16, 1)
        )
        self.out1 =nn.Sequential(
            nn.Linear(32, 16),
            nn.BatchNorm1d(1),
            nn.Sigmoid(),
            nn.Linear(16, 1)
        )
        self.out2 = nn.Sequential(
            nn.Linear(64, 16),
            nn.BatchNorm1d(1),
            nn.Sigmoid(),
            nn.Linear(16, 1)
        )

        # self.linear = nn.Sequential(
        #     nn.Linear(7, 3),
        #     nn.BatchNorm1d(1),
        #     nn.ReLU(),
        #     nn.Linear(3, 1)
        # )

    def forward(self, x):
        r_out, h_state = self.rnn(x, None)  #
        r_out, h_state = self.rnn1(r_out, None)
        x1=r_out[:, -1, :]
        x2 = self.out(x1.reshape(x1.shape[0], 1, x1.shape[1]))
        for i in range(6):
            x2 = torch.cat((x2, x2), 1)

        x2 = x2.permute(0, 2, 1)

        r_out, h_state = self.rnn2(x2, None)
        r_out, h_state = self.rnn3(r_out, None)
        x1 = r_out[:, -1, :]
        x2 = self.out1(x1.reshape(x1.shape[0], 1, x1.shape[1]))
        out = x2.squeeze(-1)

        for i in range(8):
            x2 = torch.cat((x2, x2), 1)

        x2 = x2.permute(0, 2, 1)
        r_out, h_state = self.rnn4(x2, None)
        r_out, h_state = self.rnn5(r_out, None)
        x1 = r_out[:, -1, :]
        x2 = self.out2(x1.reshape(x1.shape[0], 1, x1.shape[1]))
        out = x2.squeeze(-1)

        return out

    # def forward(self, x):
    #     r_out, h_state = self.rnn(x, None)  #
    #     r_out, h_state = self.rnn1(r_out, None)
    #
    #     outs = []
    #     for time in range(r_out.size(1)):
    #         outs.append(self.out(r_out[:, time, :]))
    #     x= torch.stack(outs, dim=-1)
    #     x = self.linear(x)
    #
    #     for i in range(6):
    #         x=torch.cat((x,x),1)
    #     x=x.permute(0,2,1)
    #
    #     r_out, h_state = self.rnn2(x, None)
    #     r_out,  h_state = self.rnn3(r_out, None)
    #
    #     outs = []
    #     for time in range(r_out.size(1)):
    #         outs.append(self.out1(r_out[:, time, :]))
    #     x = torch.stack(outs, dim=-1)
    #     out=x.squeeze(-1)
    #     return out

