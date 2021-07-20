import numpy as np

import ggln

import matplotlib.pyplot as plt
import torch

ggln1 = ggln.GGLN([8,8,8,8, 1], 1, 8)
ggln1 = ggln.GGLN([16,64,64, 1], 1, 3)

# xx = torch.randn((1,1))
# print(ggln1.context_function(xx ,1 ,2))

# print(ggln1.predict(xx))


# print(ggln1.predict(xx, target=torch.randn((2,))))

n =128*1
batch_size = 4
x_all = torch.rand((n,1))*6 - 3
y_all = torch.sin(3* x_all)

for ii in range(0, n, batch_size):
    print(ii)
    xx = x_all[ii:ii+batch_size, :]
    yy = y_all[ii:ii+batch_size, :]
    ggln1.predict(xx, target=yy)


x_test = torch.linspace(-3,3,100).unsqueeze(-1)
y_out, s_sq_out = ggln1.predict(x_test)

s_out = torch.sqrt(s_sq_out)

with torch.no_grad():
    # plt.fill_between(x_test.flatten(), y_out.flatten()-s_out.flatten(), y_out.flatten()+s_out.flatten() , alpha=0.3)
    plt.plot(x_all, y_all, '.')
    plt.plot(x_test.flatten(), y_out.flatten())
    plt.show()

# xx = np.random.rand(100)*6-3
# yy= np.sin(3* xx)

# for ii in range(len(xx)):
#     print(ii)
#     ggln1.predict(np.array([[xx[ii]]]),target=yy[ii])

# for ii in range(len(xx)):
#     print(ii)
#     ggln1.predict(np.array([[xx[ii]]]),target=yy[ii])

# x1 = np.linspace(-3,3,100)
# y1=np.zeros(len(x1))
# s1=np.zeros(len(x1))
# for ii, x in enumerate(x1):
#     print(ii)
#     y_out, sig_out= ggln1.predict(np.array([[x]]))
#     y1[ii] = y_out
#     s1[ii] = sig_out
# plt.figure()
# plt.plot(x1,y1)
# plt.figure()
# plt.plot(x1,s1)
# plt.show()