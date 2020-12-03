include .env
export

test-state:
	python $(SLIMEVOLLEYGYM_PATH)/test_state.py

test-pixel:
	python $(SLIMEVOLLEYGYM_PATH)/test_pixel.py

test-atari:
	python $(SLIMEVOLLEYGYM_PATH)/test_atari.py

eval-agents:
	$(SLIMEVOLLEYGYM_PATH)/eval_agents.sh

pip-freeze:
	pip freeze > requirements.txt

pip-install-requirements:
	pip install -r requirements.txt
