from qpsolvers import solve_qp
import matplotlib.pyplot as plt
import numpy as np

steps_duration = 1
slice_per_step = 1
com_max_speed = 0.5
legs_spacing = 0.1
max_step_size = 0.075
target = np.array([0.5, 0])
legs = np.array([[-1,1], [1,1], [-1,-1], [1,-1]]) * legs_spacing
legs_target = legs + target

steps = []
flying = []
step = legs.copy()
flying.append(None)
steps.append(legs)
while np.linalg.norm(step - legs_target) > 1e-5:
    for leg in range(4):
        flying.append(leg)
        delta = legs_target[leg] - step[leg]
        if np.linalg.norm(delta) < max_step_size:
            step[leg] = legs_target[leg]
        else:
            step[leg] += max_step_size * delta / np.linalg.norm(delta)
        steps.append(step.copy())
steps.append(steps[-1])
flying.append(None)

def support_polygon(k):
    points = []
    for i in 0, 1, 3, 2:
        if flying[k] == i:
            continue
        points.append(steps[k][i])
    points.append(points[0])

    return np.array(points)


n_steps = slice_per_step * len(steps)
P = np.eye(n_steps * 2)
q = np.zeros(n_steps * 2)

ub = np.ones(n_steps * 2) * com_max_speed
lb = -ub

A = np.stack((
    np.array([1, 0] * (n_steps), dtype=float),
    np.array([0, 1] * (n_steps), dtype=float)
))
b = target

G = []
h = []
for k in range(1, len(steps)-1):
    R = np.array([[0, 1], [-1, 0]])
    margin = 0.03
    polygon = support_polygon(k)
    for n in range(len(polygon)-1):
        point1 = polygon[n].reshape(2, -1)
        point2 = polygon[n+1].reshape(2, -1)
        n = R.dot(point2 - point1) / np.linalg.norm(point2 - point1)
        C = margin + point1.T.dot(n)
        constraint = [-n[0,0], -n[1,0]]*k + [0, 0]*(n_steps-k)
        G.append(constraint)
        h.append(-float(C))
G = np.array(G)
h = np.array(h)

print(P.shape)
print(q.shape)
print(A.shape)
print(b.shape)
print(G.shape)
print(h.shape)
# exit()

sol = solve_qp(P, q, G, h, A, b).reshape(-1, 2)

com = np.array([0., 0.])
coms = []
coms.append(com.copy())
for entry in sol:
    com += entry
    coms.append(com.copy())
coms.append(coms[-1])
coms = np.array(coms)
print(coms.T)

for k in range(len(steps)):
    plt.clf()
    steps_positions = np.array(steps).reshape(-1, 2)
    plt.grid()
    plt.axis('equal')
    plt.scatter(legs.T[0], legs.T[1], color='blue')

    plt.plot(coms.T[0], coms.T[1], color='gray', marker='x')
    plt.plot([coms.T[0][k]], [coms.T[1][k]], color='orange', marker='x')

    plt.scatter(steps_positions.T[0], steps_positions.T[1], color='cyan', marker='x')
    poly = support_polygon(k)

    plt.plot(poly.T[0], poly.T[1], color='green')
    plt.scatter([poly.T[0][0]], [poly.T[1][0]], color='red')
    plt.scatter([poly.T[0][1]], [poly.T[1][1]], color='red', marker='x')

    plt.scatter(legs_target.T[0], legs_target.T[1], color='orange')
    # plt.show()
    plt.pause(0.2)
    plt.savefig('walk-%02d.png' % k)
    # input()

