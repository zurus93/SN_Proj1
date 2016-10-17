set ylabel "y"
set xlabel "x"
set autoscale
set datafile separator ","

set term png
set output "regression.png"

# style
set style line 1 linetype 10 linecolor rgb "red" linewidth 2 pointsize 0

plot trainingSet using 1:2 with lines linestyle 1 title "training function",\
    testSet using 1:2:(0.13) with circles linecolor rgb "black" title "test values"

