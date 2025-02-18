# Set the output file
set terminal pdfcairo
set output sprintf("%s", ARG2)

datfile = sprintf("%s", ARG1)

# Set the labels
set xlabel "Episode"
set ylabel "Feedback Labels"
set key top left

set boxwidth 8
set style fill solid 1 border lt -1

plot datfile using ($1-8):(($2+$10+$18+$26+$34+$42+$50+$58+$66+$74)/10) with boxes title 'label 0' lc rgb "#c5c9e9", \
    datfile using ($1):(($3+$11+$19+$27+$35+$43+$51+$59+$67+$75)/10):xtic(1) with boxes title 'label 1' lc rgb "#87cdea", \
    datfile using ($1+8):(($4+$12+$20+$28+$36+$44+$52+$60+$68+$76)/10) with boxes title 'label 0.5 ' lc rgb "#6395ed"