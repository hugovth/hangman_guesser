#!/usr/bin/env bash
ln -s .common-config/.pre-commit-config.yaml .pre-commit-config.yaml && \
ln -s .common-config/pytype.cfg pytype.cfg && \
ln -s .common-config/pylama.ini pylama.ini && \
ln -s .common-config/pytest.ini pytest.ini && \
ln -s .common-config/entrypoint.sh entrypoint.sh && \
ln -s .common-config/Makefile Makefile

echo "Successfully created symlinks!"
