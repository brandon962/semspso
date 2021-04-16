import numpy as np
import random
import math

runs = 10
iters = 500
convergence = np.zeros(iters+1)
regions = 4
particles = 20

alpha = 1
beta = 1
gamma = 1
delta = 0.1
decay = 0.7

# 0: origin, 1: fitness(0: origion , 1: personal best), 2:personal best, 3: speed
sol_types = 4
# 0: exp_value, 1:ts/tns, 2: mean, 3: best, 4: ts, 5: tns
region_types = 6
degrees = 30
grange = 30
lrange = -30
solutions = np.zeros((regions, particles, sol_types, degrees))
speed_init_rate = 0.1
global_best = np.zeros((regions, 2, degrees))
global_dif = np.zeros((regions, degrees))
global_change = np.zeros(regions)

searchers = 8
searcher_sol = np.zeros((searchers, 2), dtype=int)  # 0: groups, 1: particles
searcher_avg = np.zeros(degrees)
region_players = 2
particle_players = particles
region_exp = np.zeros((regions, region_types))
total_best = np.zeros(degrees)


def update_searcher():
    for s in range(searchers):
        region_select = random.randint(0, regions-1)
        for _ in range(region_players):
            region_temp = random.randint(0, regions-1)
            if region_exp[region_temp][0] < region_exp[region_select][0]:
                region_select = region_temp
        particle_select = random.randint(0, particles-1)
        for _ in range(particle_players):
            particle_temp = random.randint(0, particles-1)
            if solutions[region_select][particle_temp][1][0] < solutions[region_select][particle_select][1][0]:
                particle_select = particle_temp
        searcher_sol[s][0] = region_select
        searcher_sol[s][1] = particle_select
        region_exp[region_select][4] += 1
        region_exp[region_select][5] = 1
    for r in range(regions):
        if r not in [searcher_sol[s][0] for s in range(searchers)]:
            region_exp[r][4] = 1
            region_exp[r][5] += 1


def update_personal_best():
    global solutions
    for r in range(regions):
        for p in range(particles):
            if solutions[r][p][0][0] < solutions[r][p][0][1]:
                solutions[r][p][0][1] = solutions[r][p][0][0]
                solutions[r][p][2] = solutions[r][p][0]


def update_global_best():
    global solutions, global_best, total_best_fit, total_best
    for r in range(regions):
        global_change[r] = 0
        for p in range(particles):
            if solutions[r][p][1][0] < global_best[r][1][0]:
                for d in range(degrees):
                    global_dif[r][d] = abs(
                        solutions[r][p][0][d]-global_best[r][0][d])
                global_best[r][0] = solutions[r][p][0]
                global_best[r][1][0] = solutions[r][p][1][0]
                global_change[r] = 1
    for r in range(regions):
        if global_best[r][1][0] < total_best_fit:
            total_best_fit = global_best[r][1][0]
            total_best = global_best[r][0]


def update_region_exp():
    global region_exp, solutions
    for r in range(regions):
        best = solutions[r][0][1][0]
        mean = 0
        for p in range(particles):
            if solutions[r][p][1][0] < best:
                best = solutions[r][p][1][0]
            mean += solutions[r][p][1][0]
        mean /= particles
        region_exp[r][1] = region_exp[r][4]/region_exp[r][5]
        region_exp[r][2] = mean
        region_exp[r][3] = best
        region_exp[r][0] = region_exp[r][2] * \
            region_exp[r][3] * region_exp[r][4]


def sphere():
    global solutions
    for r in range(regions):
        for p in range(particles):
            sum1 = 0
            for d in range(degrees):
                sum1 += solutions[r][p][0][d]*solutions[r][p][0][d]
            solutions[r][p][1][0] = sum1


def ackley():
    global solutions
    for r in range(regions):
        for p in range(particles):
            sum1 = 0
            sum2 = 0
            for d in range(degrees):
                sum1 += solutions[r][p][0][d]*solutions[r][p][0][d]
                sum2 += math.cos(2*math.pi*solutions[r][p][0][d])
            solutions[r][p][1][0] = -20*math.exp(-0.2*math.sqrt(
                sum1/degrees)) - math.exp(sum2/degrees) + 20 + math.exp(1)


def init():
    global solutions, total_best_fit, global_best, region_exp

    total_best_fit = 999999
    for r in range(regions):
        for p in range(particles):
            solutions[r][p][0][1] = 9999999999
    for r in range(regions):
        global_best[r][1][0] = 9999999999
    for r in range(regions):
        region_exp[r][4] = 1
        region_exp[r][5] = 1

    solutions = np.zeros((regions, particles, sol_types, degrees))
    for r in range(regions):
        for p in range(particles):
            for d in range(degrees):
                solutions[r][p][0][d] = random.uniform(lrange, grange)
                solutions[r][p][2][d] = solutions[r][p][0][d]
                solutions[r][p][3][d] = random.uniform(
                    speed_init_rate*(lrange-grange), speed_init_rate*(grange-lrange))


def move_particle():
    for d in range(degrees):
        searcher_avg[d] = 0
        for s in range(searchers):
            searcher_avg[d] += solutions[searcher_sol[s]
                                         [0]][searcher_sol[s][1]][0][d]
        searcher_avg[d] /= searchers

    for r in range(regions):
        for p in range(particles):
            for d in range(degrees):
                move_speed = decay*solutions[r][p][3][d] + (alpha*random.random()*(global_best[r][0][d]-solutions[r][p][0][d])+beta*random.random()*(
                    solutions[r][p][2][d]-solutions[r][p][0][d])+gamma*random.random()*(searcher_avg[d]-solutions[r][p][0][d])+delta*(random.uniform(-global_dif[r][d], global_dif[r][d])))
                solutions[r][p][3][d] = move_speed
                solutions[r][p][0][d] += move_speed


def update():
    update_personal_best()
    update_global_best()
    update_region_exp()
    update_searcher()


if __name__ == "__main__":
    for run in range(runs):
        print("run : ", run+1)
        init()
        ackley()
        update()
        # print("iter : 0 , min : ", total_best_fit)
        convergence[0] += total_best_fit
        for iter in range(iters):
            move_particle()
            ackley()
            update()
            # print("iter : ", iter+1, ", min : ", total_best_fit)
            convergence[iter+1] += total_best_fit

    for iter in range(iters+1):
        convergence[iter] /= runs
        if iter % 50 == 0:
            print(convergence[iter])
