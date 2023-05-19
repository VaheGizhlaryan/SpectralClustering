import numpy as np
import matplotlib.pyplot as plt

PERPLEXITY = 5
g_kernel = 1
EPOCHS = 2000
LR = 200
MOMENTUM = 0.99


def k_neighbours(x, x1_index, p_or_q='p'):
    x1 = x[x1_index]
    distances = []

    for i, xi in enumerate(x):
        if i != x1_index:
            if p_or_q == 'p':
                distance = np.exp(-np.linalg.norm(x1 - xi) ** 2 / (2 * g_kernel ** 2))
            else:
                distance = (1 + np.linalg.norm(x1 - xi) ** 2) ** -1
            distances.append([i, distance])

    k_neighbours = sorted(distances, key=lambda x: x[1])
    return k_neighbours[:PERPLEXITY]


def compute_pij(x, x1_index, x2_index):
    x1 = x[x1_index]
    x2 = x[x2_index]
    num = np.exp(-np.linalg.norm(x1 - x2) ** 2) / (2 * g_kernel ** 2)
    denom = sum(i[1] for i in k_neighbours(x, x1_index, 'p'))
    return num / denom


def compute_p(x):
    table = np.zeros((x.shape[0], x.shape[0]))
    for i in range(x.shape[0]):
        for j in range(x.shape[0]):
            if i != j:
                pij = compute_pij(x, i, j)
                pji = compute_pij(x, j, i)
                table[i, j] = (pij + pji) / (2 * x.shape[0])
    return table


def compute_qij(y, y1_index, y2_index):
    y1 = y[y1_index]
    y2 = y[y2_index]
    num = (1 + np.linalg.norm(y1 - y2) ** 2) ** -1
    denom = sum(i[1] for i in k_neighbours(y, y1_index, 'q'))
    return num / denom


def compute_q(y):
    table = np.zeros((y.shape[0], y.shape[0]))
    for i in range(y.shape[0]):
        for j in range(y.shape[0]):
            if i != j:
                qij = compute_qij(y, i, j)
                table[i, j] = qij
    return table


def kl_divergence(p, q):
    total = 0
    for i in range(p.shape[0]):
        for j in range(q.shape[0]):
            if q[i, j] != 0 and p[i, j] != 0:
                total += p[i, j] * np.log(p[i, j] / q[i, j])
    return total


def gradient_descent(p, q, y):
    history = np.zeros((p.shape[0], 2, y.shape[1]))
    for iter in range(EPOCHS):
        for i in range(y.shape[0]):
            sum_value = sum(
                ((y[i] - y[j]) * (p[i, j] - q[i, j]) * (1 + np.linalg.norm(y[i] - y[j] ** 2)) ** -1)
                for j in range(y.shape[0])
            )
            y[i] -= 4 * LR * sum_value + MOMENTUM * (history[i, 1] - history[i, 0])
            history[i, 0] = history[i, 1]
            history[i, 1] = y[i]
        if iter % 100 == 0:
            q = compute_q(y)
            print(kl_divergence(p, q))
    y -= np.mean(y)
    y /= np.std(y)
    return y


x = np.random.rand(30, 3)
x = np.tile(x, (2, 1))
x[:30] *= 0.1
color = ['blue'] * 30 + ['red'] * 30

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x[:, 0], x[:, 1], x[:, 2], color=color)
plt.show()

table_p = compute_p(x)

y = x.dot(np.random.rand(x.shape[1], 2))
y -= np.mean(y)
y /= np.std(y)
table_q = compute_q(y)
y = gradient_descent(table_p, table_q, y)

plt.scatter(y[:, 0], y[:, 1], color=color)
plt.show()
