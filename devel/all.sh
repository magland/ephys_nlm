#!/bin/bash

# Run this from the root directory of the project

set -ex

pytest
pylint ephys_nlm
devel/create_api_docs.sh