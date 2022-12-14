#!/bin/bash

### AQSOL

# LSPE
python main_aqsol.py --config 'config/sat_lspe.json' --model GatedGCN --out_dir 'out/LSPE/' --dataset AQSOL --seed 0
python main_aqsol.py --config 'config/sat_lspe.json' --model GatedGCN --out_dir 'out/LSPE/' --dataset AQSOL --seed 41
python main_aqsol.py --config 'config/sat_lspe.json' --model GatedGCN --out_dir 'out/LSPE/' --dataset AQSOL --seed 95

# SAT
python main_aqsol.py --config 'config/sat_lspe.json' --model SAT --out_dir 'out/SAT/' --dataset AQSOL --seed 0
python main_aqsol.py --config 'config/sat_lspe.json' --model SAT --out_dir 'out/SAT/' --dataset AQSOL --seed 41
python main_aqsol.py --config 'config/sat_lspe.json' --model SAT --out_dir 'out/SAT/' --dataset AQSOL --seed 95

# SAT-LSPE
python main_aqsol.py --config 'config/sat_lspe.json' --model SATLSPE --out_dir 'out/SATLSPE/' --dataset AQSOL --seed 0
python main_aqsol.py --config 'config/sat_lspe.json' --model SATLSPE --out_dir 'out/SATLSPE/' --dataset AQSOL --seed 41
python main_aqsol.py --config 'config/sat_lspe.json' --model SATLSPE --out_dir 'out/SATLSPE/' --dataset AQSOL --seed 95


### ZINC 

# LSPE
python main_zinc.py --config 'config/sat_lspe.json' --model GatedGCN --out_dir 'out/LSPE/' --dataset ZINC --seed 0
python main_zinc.py --config 'config/sat_lspe.json' --model GatedGCN --out_dir 'out/LSPE/' --dataset ZINC --seed 41
python main_zinc.py --config 'config/sat_lspe.json' --model GatedGCN --out_dir 'out/LSPE/' --dataset ZINC --seed 95

# SAT
python main_zinc.py --config 'config/sat_lspe.json' --model SAT --out_dir 'out/SAT/' --dataset ZINC --seed 0
python main_zinc.py --config 'config/sat_lspe.json' --model SAT --out_dir 'out/SAT/' --dataset ZINC --seed 41
python main_zinc.py --config 'config/sat_lspe.json' --model SAT --out_dir 'out/SAT/' --dataset ZINC --seed 95

# SAT-LSPE
python main_zinc.py --config 'config/sat_lspe.json' --model SATLSPE --out_dir 'out/SATLSPE/' --dataset ZINC --seed 0
python main_zinc.py --config 'config/sat_lspe.json' --model SATLSPE --out_dir 'out/SATLSPE/' --dataset ZINC --seed 41
python main_zinc.py --config 'config/sat_lspe.json' --model SATLSPE --out_dir 'out/SATLSPE/' --dataset ZINC --seed 95