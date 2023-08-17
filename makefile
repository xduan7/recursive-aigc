SRC_DIR_NAME = src

help:
	@echo "Please use \`make <target>' where <target> is one of"
	@echo "    format:      check the format"
	@echo "    lint:        check the lint"
	@echo "    type:        check the type hints"
	@echo "    check:       run format, lint, type"

type:
	@echo ">>> Checking typing with mypy ..."
	@mypy $(SRC_DIR_NAME) || true
	@echo ""

format:
	@echo ">>> Sorting imports with isort ..."
	@isort . || true
	@echo ">>> Formatting with black ...."
	@black . || true
	@echo ""

lint:
	@echo ">>> Linting with flake8 ...."
	@flake8 $(SRC_DIR_NAME) || true
	@echo ">>> Checking unused code with vulture ...."
	@vulture --min-confidence 100 $(SRC_DIR_NAME) || true
	@echo ""

check:
	@$(MAKE) format
	@$(MAKE) lint
	@$(MAKE) type
