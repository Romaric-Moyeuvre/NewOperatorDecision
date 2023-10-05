#!make
include .env
# export $(shell sed 's/=.*//' .env)

default: help

.PHONY: help
help: # Show help for each of the Makefile recipes.
	@grep -E '^[a-zA-Z0-9 -_]+:.*#'  Makefile | sort | while read -r l; do printf "\033[1;32m$$(echo $$l | cut -f 1 -d':')\033[00m:$$(echo $$l | cut -f 2- -d'#')\n"; done

.PHONY: config_init
config_init: # initialize the configuration with current user UID GID
	@service-init.sh

.PHONY: config_git_init
config_git_init: # reset git repo to a blank new one
	@service-git-init.sh

.PHONY: docker_rebuild
docker_rebuild: draz dcb # clean and rebuild

.PHONY: docker_push
docker_push: # docker push to repository
	@echo image name : ${DOCKER_REPONAME_V} 
	@docker tag ${DOCKER_IMAGENAME_L} ${DOCKER_REPONAME_V}
	@docker push ${DOCKER_REPONAME_V}

.PHONY: docker_push_as_latest
docker_push_as_latest: # docker push to repository
	@echo image name : ${DOCKER_REPONAME_L}
	@docker tag ${DOCKER_IMAGENAME_L} ${DOCKER_REPONAME_L}
	@docker push ${DOCKER_REPONAME_L}

.PHONY: docker_rmcreds
docker_rmcreds: # docker registry, erase credentials
	@docker_logout
	@rm ~/.docker/config.json

.PHONY: docker_login
docker_login: # docker registry login
	@docker login

.PHONY: docker_logout
docker_logout:#  docker registry logout
	@docker logout

.PHONY: dc_build
dc_build:# build docker images
	@docker-compose build

.PHONY: dc_upd
dc_upd:# run containers in background
	@docker-compose up -d

.PHONY: dc_down
dc_down:# stop containers
	@docker-compose down

.PHONY: dc_ps
dc_ps:# show running containers
	@docker-compose ps

.PHONY: dc_logs
dc_logs:# show containers logs
	@docker-compose logs -f

.PHONY: dc_exec_bash
dc_exec_bash:# enter bash into python-shiny container
	@docker-compose exec -u serviceuser python-shiny /bin/bash && cd ~./app

.PHONY: docker_raz
docker_raz:# kill all containers and remove image
	@docker kill ${DOCKER_NAME}
	@docker container prune
	@docker rmi $(DOCKER_NAME) || true