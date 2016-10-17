set ylabel "Error"
set xlabel "Number of Weight Updates"
set autoscale
unset colorbox
set datafile separator ","

set term png
set output "output.png"

# style
set style line 1 linetype 10 linecolor rgb "black" linewidth 2 pointsize 0
set style line 2 linetype 2 linecolor rgb "red" linewidth 2 pointsize 0


plot input using 1:2 linestyle 1 with lines title "Test set error", \
     input using 1:3 linestyle 2 with lines title "Training set error"

