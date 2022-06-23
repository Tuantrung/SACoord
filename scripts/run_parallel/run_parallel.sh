#!/bin/bash

# use GNU parallel to run multiple repetitions and scenarios in parallel
# run from project root! (where Readme is)

printf "============================================== Start ==============================================\n"
parallel sacoord :::: algorithm_config_files.txt :::: network_files.txt :::: service_files.txt :::: sim_config_files.txt
printf "============================================== Done ===============================================\n"