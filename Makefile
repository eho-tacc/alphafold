AF2_HOME ?= /scratch/projects/tacc/bio/alphafold
SIF ?= /scratch/projects/tacc/bio/alphafold/images/alphafold_2.0.0.sif

test-short:
	singularity exec $(SIF) python3 run_alphafold_test.py

test-reduced:
	singularity exec $(SIF) python3 run_alphafold.py --flagfile=$(AF2_HOME)/test-container/flags/reduced_dbs.ff \
		--fasta_paths=$(AF2_HOME)/test-container/input/sample.fasta \
		--output_dir=$$PWD/tests/output \
		--model_names=model_1