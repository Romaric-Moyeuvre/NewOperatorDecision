#!/bin/bash
MY_UID=$(id -u)
MY_GID=$(id -g)
if [ -f ".env" ]
then
    echo env exists
else
    echo new env from example
    cp .env.example .env
fi
sed -i '/^FORCE_USER_ID=/s/=.*/='$MY_UID'/' .env
sed -i '/^FORCE_GROUP_ID=/s/=.*/='$MY_GID'/' .env
echo
echo Your ENV is now :
echo -------------------------------------
cat .env
echo -------------------------------------

