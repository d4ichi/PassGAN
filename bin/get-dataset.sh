#!/bin/sh

# This script download "rockyou-withcount.txt.bz2" and extracts it.
# File URL: http://downloads.skullsecurity.org/passwords/rockyou-withcount.txt.bz2

printf -- "\nDownloading \"rockyou-withcount.txt.bz\"...\n\n";
wget "http://downloads.skullsecurity.org/passwords/rockyou-withcount.txt.bz2" -P ../data

printf -- "Extracting  \"rockyou-withcount.txt.bz\"...";
bzip2 -d "../data/rockyou-withcount.txt.bz2"

printf -- "...Done\n\n"

exit 0;
