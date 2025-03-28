.PHONY: install setup run clean deploy check-deployment version bump-version

# Development setup
install:
	pip install -r requirements.txt

setup:
	python -m venv venv
	. venv/bin/activate && pip install -r requirements.txt

run:
	python main.py

clean:
	rm -rf __pycache__
	rm -rf venv
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

# Version management
version:
	@python -c "import version; print(f'Current version: {version.VERSION} (Built: {version.BUILD_DATE})')"

bump-version:
	@if [ -z "$(VERSION)" ]; then \
		echo "Usage: make bump-version VERSION=x.y.z"; \
	else \
		bash scripts/deploy.sh $(VERSION); \
	fi

# Deployment helpers
deploy:
	bash scripts/deploy.sh

check-deployment:
	@echo "Usage: make check-deployment IP=<instance-ip> [KEY=<ssh-key-path>]"
	@if [ -n "$(IP)" ]; then \
		if [ -n "$(KEY)" ]; then \
			python scripts/check_deployment.py --ip $(IP) --ssh-key $(KEY); \
		else \
			python scripts/check_deployment.py --ip $(IP); \
		fi \
	fi

# Default
all: install run 