import math
import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as signal
import torch
torch.cuda.current_device()
import torch.utils.data as Data
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error, mean_absolute_error
import time
from pandas import DataFrame
# from model_torch import *
# from model_copy import *
from RNN import *
# from transformer import *
import torchvision

start = time.perf_counter()


EPOCH =85
BATCH_SIZE =64
LR = 0.0003
model=Rnn(512)
# 传统LSTM
# class LSTM(nn.Module):
#     def __init__(self):
#         super(LSTM, self).__init__()
#         self.lstm = nn.LSTM(
#             input_size=6,
#             hidden_size=64,
#             num_layers=2,
#             batch_first=True,
#             dropout=0.05,
#         )
#         self.out = nn.Sequential(
#             nn.Linear(64, 10),
#             nn.BatchNorm1d(10, momentum=0.5),
#             nn.ReLU(),
#             nn.Linear(10, 1),
#         )
#
#     def forward(self, x):
#         x = x.permute(0, 2, 1).contiguous()
#         r_out, (h_n, h_c) = self.lstm(x, None)
#         out = self.out(r_out[:, -1, :])
#         return out


if torch.cuda.is_available():
     model = model.cuda()

train_x1 = np.load(r'C:\Users\Administrator\Desktop\大创\data\data_c1_512_7_2d.npy')
train_x2 = np.load(r'C:\Users\Administrator\Desktop\大创\data\data_c4_512_7_2d.npy')

train_y1 = np.load(r'C:\Users\Administrator\Desktop\大创\data\data0\data_c1_labels.npy')
train_y2 = np.load(r'C:\Users\Administrator\Desktop\大创\data\data0\data_c4_labels.npy')

test_x = np.load(r'C:\Users\Administrator\Desktop\大创\data\data_c6_512_7_2d.npy')
test_y = np.load(r'C:\Users\Administrator\Desktop\大创\data\data0\data_c6_labels.npy')

yy0 = test_y

# # fft的数据，在取模后取log
# train_x1 = abs(train_x1)
# train_x1 = np.log(train_x1)
# train_x2 = abs(train_x2)
# train_x2 = np.log(train_x2)
# test_x = abs(test_x)
# test_x = np.log(test_x)

mm = preprocessing.MinMaxScaler()
train_y1 = mm.fit_transform(train_y1.reshape(-1, 1))
train_y2 = mm.fit_transform(train_y2.reshape(-1, 1))
test_y = mm.fit_transform(test_y.reshape(-1, 1))
test_y = test_y.reshape(-1)

train_x = np.append(train_x1, train_x2, axis=0)
train_y = np.append(train_y1, train_y2, axis=0)

# 打印出训练集以及测试集数据的大小
print(train_x.shape)
print(train_y.shape)
print(test_x.shape)
print(test_y.shape)


# 将训练集数据封装进TensorDataset
train_x = torch.from_numpy(train_x)
train_x = train_x.permute(0, 2, 1)


train_y = torch.from_numpy(train_y)
train_dataset = Data.TensorDataset(train_x, train_y)

# 将测试集数据封装进TensorDataset
test_x = torch.from_numpy(test_x)
test_x = test_x.permute(0, 2, 1)


test_y = torch.from_numpy(test_y)
test_dataset = Data.TensorDataset(test_x, test_y)
test_loader = Data.DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False, )

optimizer = torch.optim.Adam(model.parameters(), lr=LR)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=500, gamma=0.9)  #######
loss_func = torch.nn.MSELoss()

train_loss = []
val_loss = []

# 训练
for epoch in range(EPOCH):

    all_num = train_x.shape[0]
    train_num = int(all_num * 0.8)
    train_data, val_data = Data.random_split(train_dataset, [train_num, all_num - train_num])

    train_loader = Data.DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True, )
    val_loader = Data.DataLoader(dataset=val_data, batch_size=BATCH_SIZE, shuffle=True, )

    total_loss = 0
    total_loss2 = 0
    model.train()
    for step, (b_x, b_y) in enumerate(train_loader):
        b_x = b_x.float()
        b_y = b_y.float()
        if torch.cuda.is_available():
            b_x = b_x.cuda()
            b_y = b_y.cuda()
        output = model(b_x)
        loss = loss_func(output, b_y)


        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        total_loss += loss.cpu().item()

    total_loss /= len(train_loader.dataset)
    train_loss.append(total_loss)

    model.eval()
    with torch.no_grad():
        for i, (v_x, v_y) in enumerate(val_loader):
            v_x = v_x.float()
            v_y = v_y.float()
            if torch.cuda.is_available():
                v_x = v_x.cuda()
                v_y = v_y.cuda()

            test_output = model(v_x)
            v_loss = loss_func(test_output, v_y)

            total_loss2 += v_loss.cpu().item()

        total_loss2 /= len(val_loader.dataset)
        val_loss.append(total_loss2)

    print('Train Epoch: {} \t Train Loss:{:.6f} \t Val Loss:{:.6f}'.format(epoch, total_loss, total_loss2))
end = time.perf_counter()
print('Running time: %s Seconds' % (end - start))

torch.save(model.state_dict(), 'checkpoint.pt')
X0 = np.array(train_loss).shape[0]
X1 = range(0, X0)
X2 = range(0, X0)
Y1 = train_loss
Y2 = val_loss
plt.subplot(2, 1, 1)
plt.plot(X1, Y1, '-')
plt.ylabel('train_loss')
plt.subplot(2, 1, 2)
plt.plot(X2, Y2, '-')
plt.ylabel('val_loss')
plt.show()


# 测试
pred = torch.empty(1)
model.eval()
model.load_state_dict(torch.load('checkpoint.pt'))
with torch.no_grad():
    for i, (tx, ty) in enumerate(test_loader):
        tx = tx.float()
        ty = ty.float()
        if torch.cuda.is_available():
            tx = tx.cuda()
            ty = ty.cuda()

        out = model(tx).squeeze(-1)
        pred = torch.cat((pred, out.cpu()))

pred = np.delete(pred.detach().numpy(), 0, axis=0)

pred = mm.inverse_transform(pred.reshape(-1, 1))


xx1 = range(0, yy0.shape[0])
yy1 = pred
yy2 = signal.savgol_filter(pred.reshape(-1), 31, 1)


# np.save(r'E:\data0\results\transformer\1c1024_2d\c1.4_6\1\1c1024_2d_yy0.npy', yy0)
# np.save(r'E:\data0\results\transformer\1c1024_2d\c1.4_6\1\1c1024_2d_yy1.npy', yy1)
# np.save(r'E:\data0\results\transformer\1c512_2d\c1.4_6\2\1c512_2d_yy2.npy', yy2)

# 求在经过不同滤波参数平滑下的MAE、RMSE值
def mae_rmse(x, y5):
    mae = mean_absolute_error(x, y5)
    rmse = math.sqrt(mean_squared_error(x, y5))
    return mae, rmse

# 求R2的值
def R2Value(x, y5):
    # y5数据类型转化
    y5 = y5.reshape(y5.shape[0], 1)
    z5 = []
    for i in range(0, len(y5)):
        for j in y5[i]:
            z5.append(j)
    y5 = np.array(z5)
    # 计算x和y5的相关系数
    data5 = DataFrame({'x': x, 'y': y5})
    r_5 = data5.x.corr(data5.y)
    r2_5 = r_5 ** 2
    return r2_5

# 计算未平滑的拟合曲线的RAE\RMSE\R2
def assess0(yy0, yy1):
    # yy2 = signal.savgol_filter(yy1.reshape(-1), sav2[0], sav2[1])
    a2 = mae_rmse(yy0, yy1)
    mae2 = a2[0]
    rmse2 = a2[1]
    r2_2 = R2Value(yy0, yy1)
    print('Test MAE: %.3f' % mae2)
    print('Test RMSE: %.3f' % rmse2)
    print('Test R2: %.3f' % r2_2)

# 计算经过不同滤波参数平滑的拟合曲线的RAE\RMSE\R2
def assess1(yy0, yy2, sav2):
    a2 = mae_rmse(yy0, yy2)
    mae2 = a2[0]
    rmse2 = a2[1]
    r2_2 = R2Value(yy0, yy2)
    print('MAE after Smoothing', sav2, ': %.3f' % mae2)
    print('RMSE after Smoothing', sav2, ': %.3f' % rmse2)
    print('R2 after Smoothing', sav2, ': %.3f' % r2_2)

# 未平滑的拟合曲线的RAE\RMSE\R2
assess0(yy0, yy1)
# 经过不同滤波参数平滑的拟合曲线的RAE\RMSE\R2
sav2 = [31,1]   ###
yy2 = signal.savgol_filter(yy1.reshape(-1), sav2[0], sav2[1])
assess1(yy0, yy2, sav2)

# 画出拟合结果
plt.plot(xx1, yy0, 'k', label='Actual Value')
plt.plot(xx1, yy1, '#929591', label='Predicted Value')
plt.plot(xx1, yy2, 'r', label='Smoothed Pre-Value' + str(sav2))

plt.xlabel('Times of cutting')
plt.ylabel(r'Average wear$\mu m$')
plt.legend(loc=4)
plt.show()
torch.save(model,r"C:\Users\Administrator\Desktop\model\1-4-6\1-4-6-02.pt")