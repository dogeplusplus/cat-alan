.PHONY:test
test:
	pytest tests/

.PHONY:install-hooks
install-hooks:
	precommit install

.PHONY: mlflow
mlflow:
	mlflow ui
