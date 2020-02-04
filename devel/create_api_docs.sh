#!/bin/bash

# Run this from the root directory of the project

set -ex

if [ -d "doc/api" ]; then rm -r doc/api; fi;
pdoc ./ephys_nlm --output-dir doc/api