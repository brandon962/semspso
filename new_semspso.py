import numpy as np
import random
import math
import pandas as pd
from functions import *
import sys 

runs = 1
iters = 10000000000000000000
region_origin = 8
regions = region_origin
particles = 25
evaluation_max = 400000
evaluation_stage = int(evaluation_max/regions)
stage = 0
stage_max = 7
convergence = np.zeros(evaluation_max+1)
evaluation = 0
fname = {0: 'Ackley', 1: 'Rastrigin', 2: 'Sphere', 3: 'Rosenbrock', 4: 'Michalewitz',
         5: 'Griewank', 6: 'Schwefel', 7: 'Sum_squares', 8: 'Zakharov', 9: 'Powell'}

func_c = 9

alpha = np.array([1.5, 1, 1, 1, 1.5, 1, 1, 1])
beta = np.array([1, 1.5, 1, 1, 1, 1.5, 1, 1])
gamma = np.array([1, 1, 1.5, 1, 1, 1, 1.5, 1])
delta = np.array([1, 1, 1, 1, 1, 1, 1, 1])
decay = np.array([0.7, 0.7, 0.7, 0.7, 0.8, 0.8, 0.8, 0.8])

# 0: origin, 1: fitness(0: origion , 1: personal best), 2:personal best, 3: speed
sol_types = 4
# 0: exp_value, 1:ts/tns, 2: mean, 3: best, 4: ts, 5: tns
region_types = 6
degrees = 30
grange = DOMAIN[fname[func_c]][1]
lrange = DOMAIN[fname[func_c]][0]
solutions = np.zeros((regions, particles, sol_types, degrees))
speed_init_rate = 0.000001
global_best = np.zeros((regions, 2, degrees))
global_dif = np.zeros((regions, degrees))
global_change = np.zeros(regions)


searchers = 8
searcher_sol = np.zeros((searchers, 2), dtype=int)  # 0: groups, 1: particles
searcher_avg = np.zeros(degrees)
region_players = 2
particle_players = int(particles/3)
region_exp = np.zeros((regions, region_types))
total_best = np.zeros(degrees)

mutation_default = (grange-lrange)/100
mutation_change = 0.98


def update_searcher():
    global regions
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
    global solutions, regions
    for r in range(regions):
        for p in range(particles):
            if solutions[r][p][1][0] < solutions[r][p][1][1]:
                solutions[r][p][1][1] = solutions[r][p][1][0].copy()
                solutions[r][p][2] = solutions[r][p][0].copy()


def update_global_best():
    global solutions, global_best, total_best_fit, total_best, regions
    for r in range(regions):
        global_change[r] = 0
        for p in range(particles):
            if solutions[r][p][1][0] < global_best[r][1][0]:
                global_best[r][0] = solutions[r][p][0].copy()
                global_best[r][1][0] = solutions[r][p][1][0].copy()
                global_change[r] = 1
    for r in range(regions):
        if global_best[r][1][0] < total_best_fit:
            total_best_fit = global_best[r][1][0].copy()
            total_best = global_best[r][0].copy()


def update_total_best(r, p):
    global solutions, global_best, total_best_fit, total_best
    if solutions[r][p][1][0] < total_best_fit:
        total_best_fit = solutions[r][p][1][0].copy()
        total_best = solutions[r][p][0].copy()


def update_region_exp():
    global region_exp, solutions, regions
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


def init():
    global solutions, total_best_fit, global_best, region_exp, evaluation, stage, region_origin, regions
    evaluation = 0
    stage = 0
    regions = region_origin
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
    global mutation_speed, global_dif, regions
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
                mutation_speed = 10**(math.log10(abs(solutions[r][p][0][d]))-0.8)
                # mutation_speed = abs(solutions[r][p][3][d])
                # mutation_speed /= global_dif[r][d]
                move_speed = decay[r]*solutions[r][p][3][d] + (1-decay[r])*(alpha[r]*random.random()*(global_best[r][0][d]-now_place)+beta[r]*random.random()*(
                    solutions[r][p][2][d]-now_place)+gamma[r]*random.random()*(searcher_avg[d]-now_place))
                if random.uniform(0, 1) < 0.3:
                    move_speed += delta[r] * \
                        (random.uniform(-mutation_speed, mutation_speed))
                solutions[r][p][3][d] = move_speed
                solutions[r][p][0][d] += move_speed


def func():
    global solutions, evaluation, regions, particles
    for r in range(regions):
        for p in range(particles):
            solutions[r][p][1][0] = globals()[fname[func_c].lower()](
                degrees, list(solutions[r][p][0]))
            evaluation += 1
            update_total_best(r, p)
            convergence[evaluation] += total_best_fit


def update():
    update_personal_best()
    update_global_best()
    update_region_exp()
    update_searcher()


def delete_solutions():
    update_region_exp()
    global stage, evaluation, evaluation_stage, alpha, beta, gamma, delta, decay, solutions, regions, global_best, region_exp
    bad_region = 0
    if evaluation > (1+stage)*evaluation_stage:
        stage += 1
        print(stage)
        for r in range(regions):
            if region_exp[r][0] > region_exp[bad_region][0]:
                bad_region = r
        alpha = np.delete(alpha, bad_region, 0)
        beta = np.delete(beta, bad_region, 0)
        gamma = np.delete(gamma, bad_region, 0)
        delta = np.delete(delta, bad_region, 0)
        decay = np.delete(decay, bad_region, 0)
        solutions = np.delete(solutions, bad_region, 0)
        global_best = np.delete(global_best, bad_region, 0)
        region_exp = np.delete(region_exp, bad_region, 0)
        regions -= 1



if __name__ == "__main__":
    if len(sys.argv) > 1:
        func_c = int(sys.argv[1])
        grange = DOMAIN[fname[func_c]][1]
        lrange = DOMAIN[fname[func_c]][0]

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
            delete_solutions()
            update()
            print("iter : ", iter+1, ", min : ", total_best_fit)
            convergence[iter+1] += total_best_fit
            mutation_default *= mutation_change
            if evaluation >= evaluation_max:
                break
        print(total_best)
        # if total_best_fit > 0.001 :
        #     print(total_best)
        #     print(mutation_speed)
        #     exit()
    for eva in range(evaluation_max+1):
        convergence[eva] /= runs
        # print(convergence[iter])
    pd.DataFrame(convergence).to_csv(
        "./output/" + fname[func_c].lower(), header=None)
