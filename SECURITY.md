# Security Policy

## Supported version

Security fixes are made on the current default branch. Historical experiment commits remain
available for provenance but are not maintained as separately supported releases.

## Reporting a vulnerability

Please do not disclose a suspected vulnerability in a public issue. Use GitHub's private
**Security → Report a vulnerability** flow when it is available. Otherwise, contact the repository
owner privately through the contact route on their GitHub profile.

Include the affected commit or version, environment, reproduction steps, potential impact, and any
suggested mitigation. Remove API keys, access tokens, private datasets, model credentials, and user
data from logs or examples. The maintainer will acknowledge the report, assess its scope, and
coordinate a fix and disclosure when appropriate.

## Scope

Reports about credential exposure, unsafe loading of untrusted artifacts, command injection,
dependency compromise, path traversal, or unintended access to local or remote data are in scope.
Ordinary model-quality differences, numerical instability in an experimental optimizer, and
resource exhaustion caused by the documented research workloads should use the regular issue
tracker unless they cross a security boundary.

This repository is research software and is not a production security boundary. Only run model
checkpoints, datasets, and configuration files from sources you trust, and use least-privilege
credentials for Hugging Face, Weights & Biases, and GitHub.
