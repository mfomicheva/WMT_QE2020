num_gpus=1

mkdir -p experiments/H2/single-lang

# into-English
for sl in si ne et ro; do
  for id in {1..5}; do
    python train.py --src ${sl} --tgt en --output_dir experiments/H2/single-lang --model xlm_roberta_large --num_gpus ${num_gpus} --num_features 5 --run_id ${id}
  done
done

# out-of-English
for tl in de zh; do
  for id in {1..5}; do
    python train.py --src en --tgt ${tl} --output_dir experiments/H2/single-lang --model xlm_roberta_large --num_gpus ${num_gpus} --num_features 5 --run_id ${id}
  done
done
