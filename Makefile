PYTHON = python
MAIN_FILE = main.py

default:

# Run the RL program with the real reward instead of an estimated reward, and create the corresponding dat files
demo:
	$(PYTHON) $(MAIN_FILE) demo

# Run the RL program with the real reward instead of an estimated reward, and create the corresponding dat files
run_real:
	$(PYTHON) $(MAIN_FILE) real

# Run the RL program with the two-valued feedback function 10 times, and create the corresponding dat files
run_noeps:
	$(PYTHON) $(MAIN_FILE) noeps

# Run the RL program with various epsilon values 10 times each, and create the corresponding dat files
run_eps:
	$(PYTHON) $(MAIN_FILE) eps

# Rule for the reward of the combined dat files
dat/combined/reward/%.dat: dat/reward/%_0.dat dat/reward/%_1.dat dat/reward/%_2.dat dat/reward/%_3.dat dat/reward/%_4.dat dat/reward/%_5.dat dat/reward/%_6.dat dat/reward/%_7.dat dat/reward/%_8.dat dat/reward/%_9.dat
	@mkdir -p $(dir $@)
	paste $^ | column -s ' ' | tr -s '\r\t' ' ' > $@

# Rule for the feedback labels of the combined dat files
dat/combined/labels/%.dat: dat/labels/%_0.dat dat/labels/%_1.dat dat/labels/%_2.dat dat/labels/%_3.dat dat/labels/%_4.dat dat/labels/%_5.dat dat/labels/%_6.dat dat/labels/%_7.dat dat/labels/%_8.dat dat/labels/%_9.dat
	@mkdir -p $(dir $@)
	paste $^ | column -s ' ' | tr -s '\r\t' ' ' > $@

# Use gnuplot to create pdf plots of the rewards from the combined dat files
plot_reward: dat/combined/reward/mainreal.dat dat/combined/reward/mainnoeps.dat dat/combined/reward/main0.dat dat/combined/reward/main10.dat dat/combined/reward/main100.dat dat/combined/reward/main1000.dat
	@mkdir -p $(dir plot/reward/)
	gnuplot -c gnuplotscripts/plot_individual.gp dat/combined/reward/mainreal.dat plot/reward/mainreal.pdf
	gnuplot -c gnuplotscripts/plot_individual.gp dat/combined/reward/mainnoeps.dat plot/reward/mainnoeps.pdf
	gnuplot -c gnuplotscripts/plot_individual.gp dat/combined/reward/main0.dat plot/reward/main0.pdf
	gnuplot -c gnuplotscripts/plot_individual.gp dat/combined/reward/main10.dat plot/reward/main10.pdf
	gnuplot -c gnuplotscripts/plot_individual.gp dat/combined/reward/main100.dat plot/reward/main100.pdf
	gnuplot -c gnuplotscripts/plot_individual.gp dat/combined/reward/main1000.dat plot/reward/main1000.pdf
	gnuplot -c gnuplotscripts/plot_individual_var.gp dat/combined/reward/mainreal.dat plot/reward/mainreal_var.pdf
	gnuplot -c gnuplotscripts/plot_individual_var.gp dat/combined/reward/mainnoeps.dat plot/reward/mainnoeps_var.pdf
	gnuplot -c gnuplotscripts/plot_individual_var.gp dat/combined/reward/main0.dat plot/reward/main0_var.pdf
	gnuplot -c gnuplotscripts/plot_individual_var.gp dat/combined/reward/main10.dat plot/reward/main10_var.pdf
	gnuplot -c gnuplotscripts/plot_individual_var.gp dat/combined/reward/main100.dat plot/reward/main100_var.pdf
	gnuplot -c gnuplotscripts/plot_individual_var.gp dat/combined/reward/main1000.dat plot/reward/main1000_var.pdf
	gnuplot -c gnuplotscripts/plot_eps_comparison.gp

# Use gnuplot to create pdf plots of the feedback labels from the combined dat files
plot_labels: dat/combined/labels/mainnoeps.dat dat/combined/labels/main0.dat dat/combined/labels/main10.dat dat/combined/labels/main100.dat dat/combined/labels/main1000.dat
	@mkdir -p $(dir plot/labels/)
	gnuplot -c gnuplotscripts/plot_labels_individual.gp dat/combined/labels/mainnoeps.dat plot/labels/mainnoeps.pdf
	gnuplot -c gnuplotscripts/plot_labels_individual.gp dat/combined/labels/main0.dat plot/labels/main0.pdf
	gnuplot -c gnuplotscripts/plot_labels_individual.gp dat/combined/labels/main10.dat plot/labels/main10.pdf
	gnuplot -c gnuplotscripts/plot_labels_individual.gp dat/combined/labels/main100.dat plot/labels/main100.pdf
	gnuplot -c gnuplotscripts/plot_labels_individual.gp dat/combined/labels/main1000.dat plot/labels/main1000.pdf
	gnuplot -c gnuplotscripts/plot_labels_individual_per.gp dat/combined/labels/mainnoeps.dat plot/labels/mainnoeps_per.pdf
	gnuplot -c gnuplotscripts/plot_labels_individual_per.gp dat/combined/labels/main0.dat plot/labels/main0_per.pdf
	gnuplot -c gnuplotscripts/plot_labels_individual_per.gp dat/combined/labels/main10.dat plot/labels/main10_per.pdf
	gnuplot -c gnuplotscripts/plot_labels_individual_per.gp dat/combined/labels/main100.dat plot/labels/main100_per.pdf
	gnuplot -c gnuplotscripts/plot_labels_individual_per.gp dat/combined/labels/main1000.dat plot/labels/main1000_per.pdf

# Remove all files in the combined dat folder and plot folder
clean:
	rm -f dat/combined/reward/*.dat
	rm -f dat/combined/labels/*.dat
	rm -f dat/plot/reward/*.pdf
	rm -f dat/plot/labels/*.pdf

# Remove all dat files in the dat folder except in the combined folder
clean_dat:
	rm -f dat/reward/*.dat
	rm -f dat/labels/*.dat


.PHONY: default demo run_real run_noeps run_noeps run_eps plot_reward plot_labels clean clean_dat