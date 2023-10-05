#!/bin/bash
PACKAGE_NAME=`cat .env | grep PACKAGE_NAME | awk -F"=" '{print $2}'`
docker-compose exec  -u serviceuser -w /home/serviceuser/${PACKAGE_NAME}/${PACKAGE_NAME} python-shiny python main-${PACKAGE_NAME}.py