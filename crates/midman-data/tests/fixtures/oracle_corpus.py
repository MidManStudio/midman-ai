#!/usr/bin/env python3
"""A synthetic corpus of secrets for comparing the scrubber with gitleaks.

    python3 oracle_corpus.py generate DIR
    python3 oracle_corpus.py compare DIR GITLEAKS_JSON OURS_TSV

`generate` writes DIR/corpus.txt and DIR/labels.json: 12 samples of each of 24 kinds (GitHub,
GitLab, AWS, Slack, Stripe, Google, Anthropic, OpenAI, npm, PyPI, SendGrid and Hugging Face
tokens, JWTs, private keys, URL passwords, bearer headers, generic assignments), each in a
realistic line and each after a `# kind` comment line. Every value is random and made up when
the script runs, so nothing in this repository looks like a credential; the corpus itself is
not committed.

`compare` reads a gitleaks JSON report
    gitleaks dir DIR/corpus.txt --report-format json --report-path REPORT --exit-code 0
and the output of
    cargo run --release -p midman-data --example scan -- lines DIR/corpus.txt > OURS_TSV
and prints, for each kind, how many of its samples each tool found, and how many findings
fall outside every sample.
"""
import base64
import collections
import json
import random
import string
import sys

SAMPLES_PER_KIND = 12
ALNUM = string.ascii_letters + string.digits


def generate(out):
    rng = random.Random(20261021)

    def alnum(n):
        return ''.join(rng.choice(ALNUM) for _ in range(n))

    def urlsafe(n):
        return ''.join(rng.choice(ALNUM + '_-') for _ in range(n))

    def base32(n):
        return ''.join(rng.choice(string.ascii_uppercase + '234567') for _ in range(n))

    def digits(n):
        return ''.join(rng.choice(string.digits) for _ in range(n))

    def base64_text(n):
        return ''.join(rng.choice(ALNUM + '+/') for _ in range(n))

    def pem(label, lines=25):
        body = '\n'.join(base64_text(64) for _ in range(lines))
        return f"-----BEGIN {label}-----\n{body}\n-----END {label}-----"

    def jwt():
        def part(obj):
            return base64.urlsafe_b64encode(json.dumps(obj).encode()).decode().rstrip('=')
        header = part({"alg": "HS256", "typ": "JWT"})
        payload = part({"sub": alnum(10), "iat": 1700000000 + rng.randint(0, 10**6), "name": "x" * 8})
        return f"{header}.{payload}.{urlsafe(43)}"

    makers = {
        'github-pat':     lambda: f"GITHUB_TOKEN=ghp_{alnum(36)}",
        'github-oauth':   lambda: f"token: gho_{alnum(36)}",
        'github-fine':    lambda: f"export GH=github_pat_{alnum(22)}_{alnum(59)}",
        'gitlab-pat':     lambda: f"GITLAB_TOKEN=glpat-{urlsafe(20)}",
        'aws-key-id':     lambda: f"aws_access_key_id = AKIA{base32(16)}",
        'aws-secret':     lambda: f"aws_secret_access_key = {base64_text(40)}",
        'slack-bot':      lambda: f"SLACK_BOT_TOKEN=xoxb-{digits(12)}-{digits(13)}-{alnum(24)}",
        'slack-webhook':  lambda: f"url: https://hooks.slack.com/services/T{alnum(8).upper()}/B{alnum(8).upper()}/{alnum(24)}",
        'stripe-live':    lambda: f"stripe.api_key = \"sk_live_{alnum(24)}\"",
        'google-api':     lambda: f"key = \"AIza{urlsafe(35)}\"",
        'anthropic':      lambda: f"ANTHROPIC_API_KEY=sk-ant-api03-{urlsafe(93)}AA",
        'openai-proj':    lambda: f"OPENAI_API_KEY=sk-proj-{urlsafe(80)}",
        'npm':            lambda: f"//registry.npmjs.org/:_authToken=npm_{alnum(36)}",
        'pypi':           lambda: f"password: pypi-AgEIcHlwaS5vcmc{urlsafe(70)}",
        'sendgrid':       lambda: f"SENDGRID_API_KEY=SG.{urlsafe(22)}.{urlsafe(43)}",
        'huggingface':    lambda: f"HF_TOKEN=hf_{alnum(34)}",
        'jwt':            lambda: f"Cookie: session={jwt()}",
        'private-key':    lambda: pem("RSA PRIVATE KEY"),
        'private-key-ec': lambda: pem("EC PRIVATE KEY", 6),
        'url-password':   lambda: f"git clone https://deploy:{alnum(16)}@github.com/acme/private-repo.git",
        'auth-bearer':    lambda: f"curl -H \"Authorization: Bearer {urlsafe(40)}\" https://api.example.com/v1/me",
        'generic-quoted': lambda: f"client_secret = \"{alnum(32)}\"",
        'generic-env':    lambda: f"DB_PASSWORD={alnum(24)}",
        'generic-json':   lambda: f"  \"api_key\": \"{alnum(40)}\",",
    }
    lines = []
    labels = []
    for name, make in makers.items():
        for _ in range(SAMPLES_PER_KIND):
            lines.append(f"# {name}")
            first = len(lines) + 1
            lines.extend(make().split("\n"))
            labels.append((name, first, len(lines)))
            lines.append("")
    with open(out + '/corpus.txt', 'w') as corpus:
        corpus.write("\n".join(lines) + "\n")
    with open(out + '/labels.json', 'w') as handle:
        json.dump(labels, handle)
    print(len(lines), "lines,", len(labels), "samples")


def compare(out, gitleaks_json, ours_tsv):
    labels = json.load(open(out + '/labels.json'))
    theirs = collections.defaultdict(set)
    for finding in json.load(open(gitleaks_json)):
        for n in range(finding['StartLine'], finding['EndLine'] + 1):
            theirs[n].add(finding['RuleID'])
    ours = collections.defaultdict(set)
    for line in open(ours_tsv):
        number, rule, _ = line.rstrip('\n').split('\t', 2)
        ours[int(number)].add(rule)
    found = collections.defaultdict(lambda: [0, 0, 0])
    for name, first, last in labels:
        lines = range(first, last + 1)
        row = found[name]
        row[0] += any(n in theirs for n in lines)
        row[1] += any(n in ours for n in lines)
        row[2] += 1
    print(f"{'kind':16} {'gitleaks':>8} {'ours':>5} {'of':>3}")
    for name, (g, o, total) in found.items():
        print(f"{name:16} {g:8} {o:5} {total:3}")
    labeled = {n for _, a, b in labels for n in range(a, b + 1)}
    print("findings of ours outside the samples:", sum(1 for n in ours if n not in labeled))
    print("findings of gitleaks outside the samples:", sum(1 for n in theirs if n not in labeled))


if __name__ == '__main__':
    if len(sys.argv) == 3 and sys.argv[1] == 'generate':
        generate(sys.argv[2])
    elif len(sys.argv) == 5 and sys.argv[1] == 'compare':
        compare(sys.argv[2], sys.argv[3], sys.argv[4])
    else:
        sys.exit(__doc__)
