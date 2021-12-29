REPO_NAME := $(shell basename `git rev-parse --show-toplevel`)
DVC_REMOTE := ${GDRIVE_FOLDER}/${REPO_NAME}


.PHONY:test
test:
	pytest tests/

.PHONY:install-hooks
install-hooks:
	precommit install

.PHONY:tensorboard
tensorboard:
	tensorboard --logdir=runs

.PHONY:dvc
dvc:
	dvc init
	dvc remote add --default gdrive ${DVC_REMOTE}
