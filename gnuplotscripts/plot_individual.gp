# Set the output file
set terminal pdfcairo
set output sprintf("%s", ARG2)

datfile = sprintf("%s", ARG1)

# Set the labels
set xlabel "Episode"
set ylabel "Reward"
set key top left
set yrange [0:500]

plot for [i=1:10] datfile using 0:(column(2*i)) with lines title sprintf("%d. Run", i), \
    datfile using 0:(sum [i=1:10] column(2*i))/10 with lines linewidth 3 title 'Average'
