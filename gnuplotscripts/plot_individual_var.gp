# Set the output file
set terminal pdfcairo
set output sprintf("%s", ARG2)

datfile = sprintf("%s", ARG1)

# Set the labels
set xlabel "Episode"
set ylabel "Reward"
set key top left
set yrange [0:500]

# Calculate the average and standard deviation
avg(x) = (sum [i=1:10] column(2*i))/10
stddev(x) = sqrt(((sum [i=1:10] (column(2*i) - avg(x))**2)/10))

plot datfile using 0:(avg(x) + stddev(x)):(avg(x) - stddev(x)) with filledcurves fill solid 0.2 title '', \
    datfile using 0:(avg(x) + stddev(x)) with lines linewidth 1 title '+1 Std. Dev.', \
    datfile using 0:(avg(x) - stddev(x)) with lines linewidth 1 title '-1 Std. Dev.', \
    datfile using 0:(avg(x)) with lines linewidth 2 title 'Average'

