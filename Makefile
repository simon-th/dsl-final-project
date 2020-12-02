SLIMEVOLLEYGYM_PATH = ./slimevolleygym

test-state:
	python $(SLIMEVOLLEYGYM_PATH)/test_state.py

test-pixel:
	python $(SLIMEVOLLEYGYM_PATH)/test_pixel.py

test-atari:
	python $(SLIMEVOLLEYGYM_PATH)/test_atari.py

pip-freeze:
	pip freeze > requirements.txt

pip-install-requirements:
	pip install -r requirements.txt
