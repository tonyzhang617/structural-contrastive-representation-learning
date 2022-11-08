# Instructions for Downloading Datasets

Download one of the datasets with label features from [The Extreme Classification Repository](http://manikvarma.org/downloads/XC/XMLRepository.html), e.g. LF-Amazon-131K, into this directory and run the following commands to uncompress the data.

```sh
unzip <DATASET ZIP FILE>
cd <DATASET PATH>
gzip -d trn.json.gz tst.json.gz lbl.json.gz
```