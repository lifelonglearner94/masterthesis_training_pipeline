.PHONY: help install train test clean

help:
	@echo "Usage:"
	@echo "  make install    Install dependencies using uv"
	@echo "  make train      Train the model (requires hydra config)"
	@echo "  make clean      Remove artifacts"

install:
	@echo "Installing dependencies with uv..."
	uv sync

# Example: make train model=titans_mac
train:
	uv run src/train.py $(args)

test:
	uv run pytest

clean:
	rm -rf logs/
	find . -type d -name "__pycache__" -exec rm -rf {} +
