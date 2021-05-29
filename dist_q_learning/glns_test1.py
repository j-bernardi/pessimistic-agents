
import numpy as np

import glns

import matplotlib.pyplot as plt
import torch

ggln1 = glns.GGLN([8,8,8,1], 2, 8, lr=1e-1, min_sigma_sq=0.5, bias_len=3,
                init_bias_weights=[None, None, None])

n =1000
batch_size = 1
# x_all = np.random.rand(n, 1)*2 -1
# y_all = np.sin(6* x_all)
radial_sine = lambda x: np.cos(np.sqrt(x[0,:]**2+ x[1,:]**2)*6)
x_all = np.random.rand(2,n)*2 -1
y_all = radial_sine(x_all)


for ii in range(0, n, batch_size):
    if ii%100==0:
        print(ii)
        # ggln1.set_bais_weights([1, 4, 1])

    xx = x_all[:, ii]
    yy = y_all[ii]
    ggln1.predict(xx, target=[yy])

print(ggln1.update_count)
print(ggln1.update_nan_count)
print(ggln1.gln_params)
print(ggln1.gln_state)

x1_test= np.linspace(-1,1,50)
x2_test= np.linspace(-1,1,50)
y_out = np.zeros((len(x1_test), len(x2_test)))
y_true =np.zeros((len(x1_test), len(x2_test)))
for i, x1 in enumerate(x1_test):
    for j, x2 in enumerate(x2_test):
        y_out[i,j] = ggln1.predict([x1,x2])
        y_true[i,j ] = radial_sine(np.array([[x1,x2]]).T)
plt.figure()
plt.pcolor(x1_test, x2_test, y_out)
plt.colorbar()

plt.figure()
plt.pcolor(x1_test, x2_test, y_true)
plt.plot(x_all[0,:],x_all[1,:],'.')
plt.colorbar()

plt.figure()
plt.pcolor(x1_test, x2_test, y_out.clip(-1,1))
plt.colorbar()

plt.show()

# x_test = np.linspace(-1,1,100).T
# y_out= [ggln1.predict([x]) for x in x_test]

# # s_out = torch.sqrt(s_sq_out)

# # with torch.no_grad():
#     # plt.fill_between(x_test.flatten(), y_out.flatten()-s_out.flatten(), y_out.flatten()+s_out.flatten() , alpha=0.3)
# plt.plot(x_all, y_all, '.')
# plt.plot(x_test, y_out)
# plt.show()

