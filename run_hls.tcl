set root [file dirname [file normalize [info script]]]
open_project -reset [file join $root build baseline]
set_top attention_head
add_files [file join $root attention.h]
add_files [file join $root attention_s1_baseline.cpp]
add_files -tb [file join $root testbench.cpp] -cflags "-Wno-unknown-pragmas"
open_solution -reset solution1
set_part {xc7a200tfbg484-1}
create_clock -period 10 -name default
csim_design
csynth_design
exit
