train:
	training/.venv/bin/python -m \
		training.flux.latent_resizer_train \
		--train_path ./data/train \
		--test_path ./data/test \
		--vae_path ../../../data/FLUX.1-dev/vae \
		--resolution 256 \
		--gradient_checkpointing \
		--batch_size 2 \
		--steps 30000 \
		--init_weights ./flux_resizer.pt

eval:
	training/.venv/bin/python -m \
		training.flux.evaluation \
		--test_path ./data/test \
		--vae_path ./data/FLUX.1-dev/vae \
		--resizer_path ./flux_resizer.pt \
		--resolution 256 \
		--scale 2 \
		--batch_size 4 \
		--resizer_only
