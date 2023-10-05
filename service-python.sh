#!/bin/bash
DOCKER_NAME=`cat .env | grep DOCKER_NAME | awk -F"=" '{print $2}'`
docker-compose exec -u serviceuser python-shiny python