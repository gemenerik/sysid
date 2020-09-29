import numpy as np
import matplotlib.pyplot as plt

plt.figure(dpi=300)
train = np.genfromtxt('results/ff/learning_rate_1_train_error_log.csv', delimiter=',')  # replace with desired logs
test = np.genfromtxt('results/ff/learning_rate_1_test_error_log.csv', delimiter=',')
# plt.plot(range(250), train, label='Train error')
plt.plot(range(len(test)), test, label='Test error, random 1')

print(min(test))
print(list(test).index(min(test)))
print(train[list(test).index(min(test))])
print('---')

train = np.genfromtxt('results/ff/seed_2_train_error_log.csv', delimiter=',')  # replace with desired logs
test = np.genfromtxt('results/ff/seed_2_test_error_log.csv', delimiter=',')
# plt.plot(range(250), train, label='Train error')
plt.plot(range(len(train)), test, label='Test error, random 2')

print(min(test))
print(list(test).index(min(test)))
print(train[list(test).index(min(test))])
print('---')

train = np.genfromtxt('results/ff/seed_3_train_error_log.csv', delimiter=',')  # replace with desired logs
test = np.genfromtxt('results/ff/seed_3_test_error_log.csv', delimiter=',')
# plt.plot(range(250), train, label='Train error')
plt.plot(range(len(train)), test, label='Test error, random 3')

print(min(test))
print(list(test).index(min(test)))
print(train[list(test).index(min(test))])
print('---')

plt.grid('minor')
plt.legend()
plt.xlabel('Epoch')
plt.ylabel('RMSE [-]')
plt.yscale('log')
# plt.ylim(0.001, 10)
plt.ylim(0.01, 1)
# plt.ylim(0.06, 0.065)
plt.xlim(0, 250)
plt.show()

