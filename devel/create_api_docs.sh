#!/bin/bash

set -ex

if [ -d "doc/api" ]; then rm -r doc/api; fi;
pdoc ./ephys_nlm --output-dir doc/api