# English baseline model
sbatch jobscripts/bert_en_e4_b128_baseline.sh

# English full dataset
sbatch jobscripts/bert_en_e4_b128_th00_da.sh

# Dutch baseline model
sbatch jobscripts/bert_nl_e4_b128_baseline.sh

# Dutch full dataset
sbatch jobscripts/bert_nl_e4_b128_th00_da.sh

# Dutch with weighted loss based on da score
sbatch jobscripts/bert_nl_e4_b128_th00_da_wl.sh

# Dutch filtered based on da score
sbatch jobscripts/bert_nl_e4_b128_th005_da.sh
sbatch jobscripts/bert_nl_e4_b128_th025_da.sh
sbatch jobscripts/bert_nl_e4_b128_th05_da.sh
sbatch jobscripts/bert_nl_e4_b128_th06_da.sh
sbatch jobscripts/bert_nl_e4_b128_th07_da.sh
sbatch jobscripts/bert_nl_e4_b128_th08_da.sh

# Dutch with weighted loss based on mqm score
sbatch jobscripts/bert_nl_e4_b128_th00_mqm_wl.sh

# Dutch filtered based on mqm score
sbatch jobscripts/bert_nl_e4_b128_th05_mqm.sh
sbatch jobscripts/bert_nl_e4_b128_th06_mqm.sh
sbatch jobscripts/bert_nl_e4_b128_th07_mqm.sh

# Dutch with weighted loss based on mixed score
sbatch jobscripts/bert_nl_e4_b128_th00_mix_daweight05_wl.sh

# Dutch filtered based on mixed score with different da-weights
sbatch jobscripts/bert_nl_e4_b128_th05_mix_daweight04.sh
sbatch jobscripts/bert_nl_e4_b128_th05_mix_daweight05.sh
sbatch jobscripts/bert_nl_e4_b128_th05_mix_daweight06.sh
sbatch jobscripts/bert_nl_e4_b128_th06_mix_daweight04.sh
sbatch jobscripts/bert_nl_e4_b128_th06_mix_daweight05.sh
sbatch jobscripts/bert_nl_e4_b128_th06_mix_daweight06.sh
sbatch jobscripts/bert_nl_e4_b128_th07_mix_daweight04.sh
sbatch jobscripts/bert_nl_e4_b128_th07_mix_daweight05.sh
sbatch jobscripts/bert_nl_e4_b128_th07_mix_daweight06.sh