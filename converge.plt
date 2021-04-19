reset
set term png
set output sprintf("%s_convergence.png",ARG1)
set grid
set title sprintf("%s",ARG1)
set yrange[-0:0.01]
plot sprintf("output/%s",ARG1) u 2 w l lc "red" t "SEMSO",sprintf("AB3C/%s",ARG1) u 2 w l t "AB3C",sprintf("output_pso/%s",ARG1) u 2 w l t "PSO"

reset
set term png
set output sprintf("%s_convergence2.png",ARG1)
set grid
# set title sprintf("%s",ARG1)
set yrange[-0:10]
# set xrange [3600:4000]

plot sprintf("output/%s",ARG1) u 2 w l lc "red" t "SEMSO",sprintf("AB3C/%s",ARG1) u 2 w l t "AB3C",sprintf("output_pso/%s",ARG1) u 2 w l t "PSO"