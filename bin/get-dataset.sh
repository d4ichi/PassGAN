#!/bin/sh

set -e

# Download and extract the count file from a fixed SecLists revision.

printf -- "\nDownloading \"rockyou-withcount.txt.tar.gz\"...\n\n"
wget "https://raw.githubusercontent.com/danielmiessler/SecLists/5e1dc9cc79aac54b373349e2a97bbb22f1b63bb3/Passwords/Leaked-Databases/rockyou-withcount.txt.tar.gz" -P ../data

printf -- "Extracting \"rockyou-withcount.txt.tar.gz\"..."
(
    # Refuse existing files, including symbolic links, and clean up partial output.
    if [ -e "../data/rockyou-withcount.txt" ] || [ -L "../data/rockyou-withcount.txt" ]; then
        printf -- "\n../data/rockyou-withcount.txt already exists.\n" >&2
        exit 1
    fi
    set -C
    exec 3> "../data/rockyou-withcount.txt"
    trap 'rm -f "../data/rockyou-withcount.txt"' 0
    tar -xOzf "../data/rockyou-withcount.txt.tar.gz" -- rockyou-withcount.txt >&3
    trap - 0
)
rm "../data/rockyou-withcount.txt.tar.gz"

printf -- "...Done\n\n"

exit 0
