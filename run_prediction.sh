num_gpus=1


for id in {1..5}; do
  mkdir -p experiments/H2/zero-shot/run_${id}
  for model_lp in sien neen eten roen ende enzh; do
    for test_lp in sien neen eten roen ende enzh; do
      test_sl = $(echo "${test_lp}" | head -c2)
      test_tl = $(echo "${test_lp}" | tail -c3)
      python predict.py --model_path experiments/H2/single-lang/run_${id}/model.${model_lp} \
      --model_dim 1024 --output_dir experiments/H2/zero-shot/run_${id} \
      --test_file data/${test_sl}-${test_tl}/dev.${test_lp}.df.short.tsv \
      --test_mt_file data/${test_sl}-${test_tl}/word-probas/word_probas.dev.${test_lp} \
      --test_wp_file data/${test_sl}-${test_tl}/word-probas/mt.dev.${test_lp} \
      --test_features_file data/${test_sl}-${test_tl}/${test_lp}.features.dev.tsv \
      --num_features 5 \
      --src ${test_sl} --tgt ${test_tl} \
      --model_name xlm-roberta-large
