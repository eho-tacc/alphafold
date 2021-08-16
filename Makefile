SIF ?= /scratch/projects/tacc/bio/alphafold/images/alphafold_2.0.0.sif

test:
	singularity exec $(SIF) python3 run_alphafold_test.py