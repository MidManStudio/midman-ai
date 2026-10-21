#!/usr/bin/env python3
"""Regenerates the fixtures of the midman-data tests from independent sources.

sha256_vectors.txt  digests from Python's hashlib of messages that the Rust test rebuilds
                    from the same formula (see `message` below).
spdx_ids.txt        license identifiers of the official SPDX list, with their flags.
spdx_exceptions.txt exception identifiers of the official SPDX list.

Usage: python3 generate_fixtures.py [path/to/spdx/json/dir]
The SPDX files come from https://github.com/spdx/license-list-data (json/licenses.json and
json/exceptions.json); without an argument only the SHA-256 vectors are written.
"""
import hashlib
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


def message(length):
    """Byte j of the message of this length is (31 * j + 7 * length + 5) mod 256."""
    return bytes((31 * j + 7 * length + 5) % 256 for j in range(length))


def write_sha256():
    lengths = list(range(0, 301)) + [511, 512, 513, 1000, 4096, 65535, 65536, 100003]
    with open(os.path.join(HERE, "sha256_vectors.txt"), "w") as out:
        for n in lengths:
            out.write(f"{n} {hashlib.sha256(message(n)).hexdigest()}\n")


def write_spdx(directory):
    licenses = json.load(open(os.path.join(directory, "licenses.json")))
    exceptions = json.load(open(os.path.join(directory, "exceptions.json")))
    with open(os.path.join(HERE, "spdx_ids.txt"), "w") as out:
        out.write(f"# SPDX license list {licenses['licenseListVersion']}: id deprecated osi-approved\n")
        for lic in sorted(licenses["licenses"], key=lambda x: x["licenseId"]):
            out.write(f"{lic['licenseId']} {int(lic['isDeprecatedLicenseId'])} {int(bool(lic.get('isOsiApproved')))}\n")
    with open(os.path.join(HERE, "spdx_exceptions.txt"), "w") as out:
        out.write(f"# SPDX license list {exceptions['licenseListVersion']}: exception id\n")
        for exc in sorted(exceptions["exceptions"], key=lambda x: x["licenseExceptionId"]):
            out.write(f"{exc['licenseExceptionId']}\n")


if __name__ == "__main__":
    write_sha256()
    if len(sys.argv) > 1:
        write_spdx(sys.argv[1])
