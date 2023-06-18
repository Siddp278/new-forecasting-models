import torch
import torch.nn as nn

class ConvRNN(nn.Module):
    def __init__(self, model_type, input_dim, timesteps, output_dim, kernel_size1=7, kernel_size2=5, kernel_size3=3, 
                 n_channels1=32, n_channels2=32, n_channels3=32, n_units1=32, n_units2=32, n_units3=32):
        super().__init__()

        self.model_type = model_type
        self.avg_pool1 = nn.AvgPool1d(2, 2)
        self.avg_pool2 = nn.AvgPool1d(4, 4)
        self.conv11 = nn.Conv1d(input_dim, n_channels1, kernel_size=kernel_size1)
        self.conv12 = nn.Conv1d(n_channels1, n_channels1, kernel_size=kernel_size1)
        self.conv21 = nn.Conv1d(input_dim, n_channels2, kernel_size=kernel_size2)
        self.conv22 = nn.Conv1d(n_channels2, n_channels2, kernel_size=kernel_size2)
        self.conv31 = nn.Conv1d(input_dim, n_channels3, kernel_size=kernel_size3)
        self.conv32 = nn.Conv1d(n_channels3, n_channels3, kernel_size=kernel_size3)
        if model_type == "RNN":
            self.rnn1 = nn.RNN(n_channels1, n_units1, batch_first=True)
            self.rnn2 = nn.RNN(n_channels2, n_units2, batch_first=True)
            self.rnn3 = nn.RNN(n_channels3, n_units3, batch_first=True)

        if model_type == "GRU":
            self.rnn1 = nn.GRU(n_channels1, n_units1, batch_first=True)
            self.rnn2 = nn.GRU(n_channels2, n_units2, batch_first=True)
            self.rnn3 = nn.GRU(n_channels3, n_units3, batch_first=True)

        if model_type == "LSTM":
            self.rnn1 = nn.LSTM(n_channels1, n_units1, batch_first=True)
            self.rnn2 = nn.LSTM(n_channels2, n_units2, batch_first=True)
            self.rnn3 = nn.LSTM(n_channels3, n_units3, batch_first=True)    

        self.linear1 = nn.Linear(n_units1+n_units2+n_units3, output_dim)
        self.linear2 = nn.Linear(input_dim*timesteps, output_dim)
        self.zp11 = nn.ConstantPad1d(((kernel_size1-1), 0), 0)
        # zp11 output is padding_left + original tensor + padding_right (kernel_size1-1), 0) = (padding_left, padding_right), padded with 0.
        self.zp12 = nn.ConstantPad1d(((kernel_size1-1), 0), 0)
        self.zp21 = nn.ConstantPad1d(((kernel_size2-1), 0), 0)
        self.zp22 = nn.ConstantPad1d(((kernel_size2-1), 0), 0)
        self.zp31 = nn.ConstantPad1d(((kernel_size3-1), 0), 0)
        self.zp32 = nn.ConstantPad1d(((kernel_size3-1), 0), 0)
        
    def forward(self, x):
        # print(f"first input shape: {x.size()}")
        x = x.permute(0, 2, 1)
        # print(f"permute input shape: {x.size()}")
        # line1
        y1 = self.zp11(x)
        # print(f"padded input shape: {y1.size()}")
        y1 = torch.relu(self.conv11(y1))
        # print(f"padded and relu input shape: {y1.size()}")
        y1 = self.zp12(y1)
        y1 = torch.relu(self.conv12(y1))
        y1 = y1.permute(0, 2, 1)
        if self.model_type == "LSTM":
            out, (h1, c1) = self.rnn1(y1) 
        else:       
            out, h1 = self.rnn1(y1)
        # line2
        y2 = self.avg_pool1(x)
        y2 = self.zp21(y2)
        y2 = torch.relu(self.conv21(y2))
        y2 = self.zp22(y2)
        y2 = torch.relu(self.conv22(y2))
        y2 = y2.permute(0, 2, 1)
        if self.model_type == "LSTM":
            out, (h2, c2) = self.rnn2(y2) 
        else:       
            out, h2 = self.rnn2(y2)
        # line3 
        y3 = self.avg_pool2(x)
        y3 = self.zp31(y3)
        y3 = torch.relu(self.conv31(y3))
        y3 = self.zp32(y3)
        y3 = torch.relu(self.conv32(y3))
        y3 = y3.permute(0, 2, 1)
        if self.model_type == "LSTM":
            out, (h3, c3) = self.rnn3(y3) 
        else:       
            out, h3 = self.rnn3(y3)
        h = torch.cat([h1[-1], h2[-1], h3[-1]], dim=1)
        out1 = self.linear1(h)
        out2 = self.linear2(x.contiguous().view(x.shape[0], -1))
        out = out1 + out2
        return out
    
    def compute_l1(self, w):
        return torch.abs(w).sum()
    
    def compute_l2(self, w):
      return torch.square(w).sum()
    
    