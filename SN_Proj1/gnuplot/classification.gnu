set ylabel "y"
set xlabel "x"
set autoscale
set datafile separator ","

set term png
set output "classification.png"
# style
unset colorbox
set palette defined ( 1 "#ffcccc",\
                      2 "#ccffcc",\
                      3 "#ccccff",\
					  4 "#cccccc",\
					  5 "red",\
                      6 "green",\
                      7 "blue",\
					  8 "black")

plot testSet using 1:2:3 with points palette ps 2 pointtype 7 title "test values",\
	trainingSet using 1:2:($3 + 4) with points palette ps 1 pointtype 7 title "training data"

