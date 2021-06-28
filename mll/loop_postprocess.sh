# process_dirs input not yet ready
#gam_spec/d3_long Job2_unc0.0
#gam_spec/d4_long Job2_unc0.0
#gam_spec/d5_long Job2_unc0.0
#gam_spec/d6_long Job2_unc0.0
#gam_spec/d1_auto Job2_unc0.0
#gam_spec/d2_auto Job2_unc0.0


# process_dirs done
#nuc_conc/sfco Job0_unc0.01_impnull
#nuc_conc/sfco Job1_unc0.01_0null
#nuc_conc/rxtr-type pwr
#nuc_conc/rxtr-type bwr
#nuc_conc/rxtr-type phwr
#gam_spec/act7 Job2_unc0.01
#gam_spec/act12 Job2_unc0.01
#gam_spec/act32 Job2_unc0.01
#nuc_conc/nuc29 Job0_unc0.01
#nuc_conc/nuc29 Job1_unc0.05
#nuc_conc/nuc29 Job2_unc0.1
#nuc_conc/nuc29 Job3_unc0.15
#nuc_conc/nuc29 Job4_unc0.2
#gam_spec/d1_short Job2_unc0.0
#gam_spec/d2_short Job2_unc0.0
#gam_spec/d3_short Job2_unc0.0
#gam_spec/d4_short Job2_unc0.0
#gam_spec/d5_short Job2_unc0.0
#gam_spec/d6_short Job2_unc0.0

#gam_spec/d3_auto Job2_unc0.0
#gam_spec/d4_auto Job2_unc0.0
#gam_spec/d5_auto Job2_unc0.0
#gam_spec/d6_auto Job2_unc0.0
#gam_spec/d1_long Job2_unc0.0
#gam_spec/d2_long Job2_unc0.0

while read p; do
    time ./htc_postprocess.py $p
done <process_dirs.txt
