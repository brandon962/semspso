from new_semspso import *


p_sol = np.zeros((10,5,8))
p_v = np.zeros((10,5,8))
p_p = np.zeros((10,5,8))
p_pf = np.zeros(10)
p_g = np.zeros((5,8))
p_gf = 0
fit = np.zeros(10)

for i in range(10):
    for j in range(4):
        for k in range(8):
            p_sol[i][j][k] = random.random()*2
            p_p[i][j][k] = p_sol[i][j][k].copy()
    for k in range(8):
        p_sol[i][4][k] = random.random()
        p_p[i][j][k] = p_sol[i][4][k].copy()

    fit[i] = pso(p_sol[i][0],p_sol[i][1],p_sol[i][2],p_sol[i][3],p_sol[i][4])
    p_pf[i] = fit[i].copy()
 
p_g = p_sol[np.argmin(fit)].copy()
p_gf = fit[np.argmin(fit)].copy()



a = 1.5
b = 1.5
d = 0.7 

for t in range(1000):
    for i in range(10):
        for j in range(4):
            for k in range(8):
                p_v[i][j][k] = p_v[i][j][k] * d + a*random.random()*(p_p[i][j][k]-p_sol[i][j][k])+b*random.random()*(p_g[j][k]-p_sol[i][j][k])
                p_sol[i][j][k] += p_v[i][j][k]
        
        fit[i] = pso(p_sol[i][0],p_sol[i][1],p_sol[i][2],p_sol[i][3],p_sol[i][4])
        if fit[i] < p_pf[i]:
            p_pf[i] =  fit[i]
            p_p[i] = p_sol[i].copy()
    
    temp = fit[np.argmin(fit)].copy()
    if temp < p_gf:
        p_g = p_sol[np.argmin(fit)].copy()
        p_gf = fit[np.argmin(fit)].copy()

print(p_g)
print()
print(p_gf)