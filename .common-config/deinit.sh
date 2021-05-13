#!/usr/bin/env bash
git submodule deinit -f -- .common-config
rm -rf .git/modules/.common-config
git rm -f .common-config

rm -rf .pre-commit-config.yaml pylama.ini pytest.ini pytype.cfg Makefile entrypoint.sh && \
    echo "Successfully deleted symlinks."
