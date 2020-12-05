"""
Represent each cockroach as D dimensional space where D is the number of cities

Chase-swarming behavior:
    P(i) : local optimum
    Pg : global optimum

    Equation (1)
    X'(i) = X(i) + step * rand * (P(i) - X(i)) if X(i) \neq P(i)
    else
    X'(i) = X(i) + step * rand * (Pg - X(i))

    step is fixed
    rand \in [0, 1]

    Equation (2)
    P(i) = Opt_j{X(j) ||X(i) - X(j)| \leq visual, j = 1, 2, ..., N}
    Equation (3)
    Pg = Opt_i{X(i), i = 1, 2, ..., N}

Dispersing:
    Equation (4)
    X'(i) = permute(X(i))

Ruthless:
    stronger eat weaker
    Equation (5)
    X(k) = Pg
    k random integer \in [1,N]

From ZhaoHui et. al. "Cockroach Swarm Optimization"

Step 1:
Initialize swarm and parameters as CSO
    population size N
    generate cockroach swarm in region
Step 2:
    Search for P(i) and Pg by : (2) and (3)
Step 3:
    Carry out chase-swarming by (1) X(i) <- X'(i), Update Pg
Step 4:
    Disperse by equation 4
    If new X'(i) better than X(i) update X(i), otherwise return X(i), Update Pg
Step 5:
    Ruthless by (5), update Pg
Step 6:
    Do we terminate? else step 2

"""
import numpy as np
import random
import math
import functools



class CSO:

    def __init__(self, cities, N = 150, step = 10, visual = 750, end_cond = 5, iterations = 100):
        self.cities = cities
        self.D = len(cities) #number of cities
        self.N = N #population size

        self.step = step
        self.visual = visual
        self.end_cond = end_cond
        self.iterations = iterations

        # STEP 1:
        self.X = np.array([np.random.permutation(self.D) for i in range(N)], dtype='int32') #population
        self.Pi = []
        self.Pg = np.array([], dtype='int32')


    def eucl(self, x, y):
        return math.sqrt((y[1] - x[1])**2 + (y[0] - x[0])**2)

    def eval(self, x):
        total_dist = 0
        for i in range(len(x)-1):
            total_dist += self.eucl(self.cities[x[i]], self.cities[x[i+1]])
        return total_dist + self.eucl(self.cities[x[0]], self.cities[x[-1]])

    def nperm(self, l1, l2, n):
        perm_count = 0
        while perm_count < n and not functools.reduce(lambda i, j : i and j, map(lambda m, k: m == k, l1, l2), True):
            old_index, switch_elem = [(i,x) for i, x in enumerate(l1) if x != l2[i]][0]
            new_index, = np.where(l2==switch_elem)
            l1[old_index], l1[new_index] = l1[new_index], l1[old_index]
            perm_count += 1
        return l1

    def norm(self, x, y):
        total = 0
        for a in range(self.D):
            total += self.eucl(self.cities[x[a]], self.cities[y[a]])
        return total

    def cso_run(self):
        prev = np.zeros(self.D, dtype='int32')
        c = 0
        while c < self.iterations:
            if c % 100 == 0:
                print(c)
            #STEP 2
            # Equation (2)
            # P(i) = Opt_j{X(j) ||X(i) - X(j)| \leq visual, j = 1, 2, ..., N}
            # Equation (3)
            # Pg = Opt_i{X(i), i = 1, 2, ..., N}
            for i, xi in enumerate(self.X):
                local_opt = []
                for xj in self.X:
                    if self.norm(xj, xi) <= self.visual:
                        local_opt.append((xj, self.eval(xj)))
                try:
                    self.Pi[i] = min(local_opt, key = lambda x: x[1])[0]
                except:
                    self.Pi.append(min(local_opt, key = lambda x: x[1])[0])

            self.Pg = min(self.X, key = lambda x: self.eval(x))

            #STEP 3
            #Adopt some elements of a better path/the optimal path
            #Equation (1)
            # X'(i) = X(i) at most step num of elements from P(i) if X(i) \neq P(i)
            # else
            # X'(i) = X(i) at most step num of elements from P(i)
            for i, xi in enumerate(self.X):
                rand = np.random.random()
                num_perm = int(self.step * rand)
                if self.eval(xi) != self.eval(list(self.Pi[i])):
                    self.X[i] = self.nperm(xi, self.Pi[i], num_perm)
                else:
                    self.X[i] = self.nperm(xi, self.Pg, num_perm)

                #STEP 4
                # Equation (4)
                #Randomly scatter
                # X'(i) = permute(X(i)) permute a random segment to see if the value is better
                rand1 = np.random.randint(0,self.D)
                rand2 = np.random.randint(0, 30-rand1)
                scattered = np.concatenate([xi[:rand1], np.random.permutation(xi[rand1:rand1+ rand2]), xi[rand1+rand2:]])
                if self.eval(scattered) > self.eval(xi):
                    self.X[i] = scattered

            self.Pg = min(self.X, key = lambda x: self.eval(x))

            #STEP 5
            # Equation (5)
            # X(k) = Pg
            # k random integer \in [1,N]
            k = random.sample(range(self.N), 1)
            self.X[k] = self.Pg

            #STEP 6:
            #Check termination
            c += 1

        return self.Pg

#oliver30_nodes = {0: (54,67), 1: (54,62), 2: (37, 84), 3: (41, 94), 4: (2, 99), 5: (7, 64), 6: (25, 62), 7: (22, 60), 8: (18, 54), 9: (4, 50), \
                  #10: (13,40), 11: (18,40), 12: (24,42), 13: (25,38), 14: (44,35), 15: (41,26), 16: (45,21), 17: (58,35), 18: (62,32), 19: (82,7), \
                  #20: (91,38), 21: (83,46), 22: (71,44), 23: (64,60), 24: (68,58), 25: (83,69), 26: (87,76), 27: (74,78), 28: (71,71), 29: (58,69)}

#cso = CSO(oliver30_nodes)
#print(cso.eval(cso.cso_run()))
