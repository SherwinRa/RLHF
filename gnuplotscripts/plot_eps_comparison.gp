# Set the output file
set terminal pdfcairo
set output "plot/reward/eps_comparison.pdf"

datfile = sprintf("%s", ARG1)

# Set the labels
set xlabel "Episode"
set ylabel "Reward"
set key top left
set yrange [0:500]

filename1 = 'dat/combined/reward/mainreal.dat'
filename2 = 'dat/combined/reward/mainnoeps.dat'
filename3 = 'dat/combined/reward/main0.dat'
filename4 = 'dat/combined/reward/main10.dat'
filename5 = 'dat/combined/reward/main100.dat'
filename6 = 'dat/combined/reward/main1000.dat'

# Calculate the average and standard deviation
avg(x) = (sum [i=1:10] column(2*i))/10

plot filename1 using 0:(avg(x)) with lines linewidth 2 title 'Real Reward', \
    filename2 using 0:(avg(x)) with lines linewidth 2 title 'No Eps', \
    filename3 using 0:(avg(x)) with lines linewidth 2 title 'Eps=0', \
    filename4 using 0:(avg(x)) with lines linewidth 2 title 'Eps=10', \
    filename5 using 0:(avg(x)) with lines linewidth 2 title 'Eps=100', \
    filename6 using 0:(avg(x)) with lines linewidth 2 title 'Eps=1000'

