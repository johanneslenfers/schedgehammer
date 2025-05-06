#!/bin/bash

set -ex

mkdir -p test_data
pushd test_data

wget https://www.cise.ufl.edu/research/sparse/MM/Williams/webbase-1M.tar.gz \
  https://s3.us-east-2.amazonaws.com/frostt/frostt_data/nell/nell-2.tns.gz \
  https://www.cise.ufl.edu/research/sparse/MM/Boeing/pwtk.tar.gz

gzip -d ./*.gz
find . -name "*.tar" -exec tar xfv {} \;

mv pwtk/pwtk.mtx .
mv webbase-1M/webbase-1M.mtx .

rmdir pwtk
rmdir webbase-1M
rm ./*.tar

popd
