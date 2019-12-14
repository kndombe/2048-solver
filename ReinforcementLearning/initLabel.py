import numpy as np

DISCOUNT = 0.9

if __name__ == "__main__":
    r = np.genfromtxt('EpisodeLogs/rate2corrected_ar', delimiter=',')[:, 1]
    Q = np.zeros_like(r)
    for i in range(r.size-1, -1, -1):
        if r[i] < 0:
            Q[i] = 0
        else:
            Q[i] = r[i] + DISCOUNT * Q[i+1]
    np.save("Qvalues/initQ_complex", Q)

    r[np.where(r > 0)] = 1
    r[np.where(r < 0)] = -3
    for i in range(r.size-1, -1, -1):
        if r[i] < 0:
            Q[i] = r[i]
        else:
            Q[i] = r[i] + DISCOUNT * Q[i+1]
    np.save("Qvalues/initQ_simple", Q)
