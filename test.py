import numpy as np
from functions import *


a = np.array([[1,2,3,4],[2,4,6,8],[3,6,9,12]])

a = np.delete(a,1,0)

print(a[1])

# b = np.random.rand(3)*100

# print(b)
fname = {0: 'Ackley', 1: 'Rastrigin', 2: 'Sphere', 3: 'Rosenbrock', 4: 'Michalewitz',
         5: 'Griewank', 6: 'Schwefel', 7: 'Sum_squares', 8: 'Zakharov', 9: 'Powell'}

func_c = 4
func_name = fname[func_c]

print(DOMAIN[fname[func_c]][0])