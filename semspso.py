import numpy as np
import random
import math

runs = 1
iters = 1000
convergence = np.zeros(iters+1)
regions = 4
particles = 50

func_c = 3
alpha = 1
beta = 1
gamma = 1
delta = 1
decay = 0.7

# 0: origin, 1: fitness(0: origion , 1: personal best), 2:personal best, 3: speed
sol_types = 4
# 0: exp_value, 1:ts/tns, 2: mean, 3: best, 4: ts, 5: tns
region_types = 6
degrees = 30
grange = 10
lrange = -10
solutions = np.zeros((regions, particles, sol_types, degrees))
speed_init_rate = 0.00001
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

mutation_default = (grange-lrange)/100
mutation_change = 0.98


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
            if solutions[r][p][1][0] < solutions[r][p][1][1]:
                solutions[r][p][1][1] = solutions[r][p][1][0]
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

def michalewicz():
    for r in range(regions):
        for p in range(particles):
            sum1 = 0
            sum2 = 0
            for d in range(degrees):
                sum1 -= math.sin(solutions[r][p][0][d]) * math.pow(math.sin(((d+1)*math.pow(solutions[r][p][0][d],2)/math.pi)),20)
            solutions[r][p][1][0] = sum1

def rastrigin() :
    global solutions
    for r in range(regions):
        for p in range(particles):
            sum1 = 0
            sum2 = 0
            for d in range(degrees):
                sum1 += (solutions[r][p][0][d]) ** 2
                sum2 += math.cos(2*math.pi*(solutions[r][p][0][d]))
            solutions[r][p][1][0] = 10*degrees + sum1 - 10 *sum2

def rosenbrock():
    global solutions
    for r in range(regions):
        for p in range(particles):
            sum1 = 0
            sum2 = 0
            for d in range(degrees-1):
                sum1 += (solutions[r][p][0][d+1]-(solutions[r][p][0][d]*solutions[r][p][0][d]))*(solutions[r][p][0][d+1]-(solutions[r][p][0][d]*solutions[r][p][0][d]))
                sum2 += (solutions[r][p][0][d]-1)*(solutions[r][p][0][d]-1)
            solutions[r][p][1][0] = 100*sum1 + sum2

def init():
    global solutions, total_best_fit, global_best, region_exp

    solutions = np.zeros((regions, particles, sol_types, degrees))
    total_best_fit = 999999
    for r in range(regions):
        for p in range(particles):
            solutions[r][p][1][1] = 9999999999
    for r in range(regions):
        global_best[r][1][0] = 9999999999
    for r in range(regions):
        region_exp[r][4] = 1
        region_exp[r][5] = 1

    for r in range(regions):
        for p in range(particles):
            for d in range(degrees):
                solutions[r][p][0][d] = random.uniform(lrange, grange)
                solutions[r][p][2][d] = solutions[r][p][0][d]
                solutions[r][p][3][d] = random.uniform(
                    speed_init_rate*(lrange-grange), speed_init_rate*(grange-lrange))


def move_particle():
    global mutation_speed, global_dif
    for d in range(degrees):
        searcher_avg[d] = 0
        for s in range(searchers):
            searcher_avg[d] += solutions[searcher_sol[s]
                                         [0]][searcher_sol[s][1]][0][d]
        searcher_avg[d] /= searchers

    for r in range(regions):
        mutation_speed = 0
        # for p in range(particles):
        #     for d in range(degrees):
        #         if abs(total_best[d]-int(total_best[d])) > mutation_speed:
        #             mutation_speed = abs(total_best[d]-int(total_best[d]))
        for p in range(particles):
            for d in range(degrees):
                now_place = solutions[r][p][0][d]
                # mutation_speed = abs(solutions[r][random.randint(0,particles-1)][0][d] - solutions[r][random.randint(0,particles-1)][0][d])
                # print(global_dif[r][d])
                # if mutation_speed > 0.01 and mutation_speed < 0.5:
                    # mutation_speed = 1 - mutation_speed
                mutation_speed = abs(solutions[r][p][0][d]-int(solutions[r][p][0][d]))
                # mutation_speed = abs(solutions[r][p][3][d])
                # mutation_speed /= global_dif[r][d]
                move_speed = decay*solutions[r][p][3][d] + (1-decay)*(alpha*random.random()*(global_best[r][0][d]-now_place)+beta*random.random()*(
                    solutions[r][p][2][d]-now_place)+gamma*random.random()*(searcher_avg[d]-now_place))
                if random.uniform(0,1) < 0.5:
                    move_speed += delta*(random.uniform(-mutation_speed, mutation_speed))
                solutions[r][p][3][d] = move_speed
                solutions[r][p][0][d] += move_speed


def func():

    # solutions[0][0][0][0] = -4.97530916
    # solutions[0][0][0][1] = -4.97530916
    # michalewicz()
    # print(solutions[0][0][1][0])
    # exit()


    if func_c == 0:
        ackley()
    elif func_c == 1:
        sphere()
    elif func_c == 2:
        rastrigin()
    elif func_c == 3:
        rosenbrock()
    elif func_c == 4:
        michalewicz()
    else:
        ackley()

def update():
    update_personal_best()
    update_global_best()
    update_region_exp()
    update_searcher()


if __name__ == "__main__":
    for run in range(runs):
        print("run : ", run+1)
        init()
        func()
        update()
        # print("iter : 0 , min : ", total_best_fit)
        convergence[0] += total_best_fit
        for iter in range(iters):
            move_particle()
            func()
            update()
            print("iter : ", iter+1, ", min : ", total_best_fit)
            convergence[iter+1] += total_best_fit
            mutation_default *= mutation_change
        print(total_best)
        # if total_best_fit > 0.001 :
        #     print(total_best)
        #     print(mutation_speed)
        #     exit()
    for iter in range(iters+1):
        convergence[iter] /= runs
        # print(convergence[iter])
