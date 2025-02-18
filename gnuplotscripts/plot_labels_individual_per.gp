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

plot datfile using ($1-8):(($6+$14+$22+$30+$38+$46+$54+$62+$70+$78)/10) with boxes title 'label 0' lc rgb "#c5c9e9", \
    datfile using ($1):(($7+$15+$23+$31+$39+$47+$55+$63+$71+$79)/10):xtic(1) with boxes title 'label 1' lc rgb "#87cdea", \
    datfile using ($1+8):(($8+$16+$24+$32+$40+$48+$56+$64+$72+$80)/10) with boxes title 'label 0.5 ' lc rgb "#6395ed"