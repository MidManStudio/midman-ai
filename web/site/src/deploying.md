# Deploying

The site is deployed to Cloudflare Pages by the `deploy-site.yml` workflow, **by hand only**: it never runs on a push. It builds the WebAssembly engine, runs the tests, builds this book, pulls the latest CI, test and benchmark results, and publishes.

**Setup:** the repository needs `CLOUDFLARE_API_TOKEN` and `CLOUDFLARE_ACCOUNT_ID` secrets. The Pages project is created on the first run.

**Status:** written; its first run on GitHub has not happened yet.
