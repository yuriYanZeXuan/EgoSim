# Egodex — quick smoke test with bundled mini samples
PYTHONPATH=. python egowm/inference/runner.py \
  --dataset egodex \
  --model_root ./EgoSim-14B \
  --dataset_root tests/samples/demo_data/egodex \
  --metadata_path tests/samples/demo_data/egodex_metadata.csv \
  --output_dir output_egodex \
  --num_inference_steps 50 \
  --gpu_id 0

# EgoVid — quick smoke test with bundled mini samples
PYTHONPATH=. python egowm/inference/runner.py \
  --dataset egovid \
  --model_root ./EgoSim-14B \
  --dataset_root tests/samples/demo_data/egovid \
  --metadata_path tests/samples/demo_data/egovid_metadata.csv \
  --output_dir output_egovid \
  --num_inference_steps 50 \
  --gpu_id 0