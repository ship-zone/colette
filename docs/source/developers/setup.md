# Setup

This describes setup from sources, mostly for developers and advanced users

1. Clone the repository

```bash
git clone https://github.com/jolibrain/colette.git
```

2. Install dependencies and CLI

```bash
pip install -e .
```

3. Run the server
```bash
./server/run.sh
```
If this succeeds, the swagger API is then available from `http://localhost:1873/docs`

Create the service via the REST API:

```bash
curl -X PUT http://localhost:1873/v1/app/app_colette -H "Content-Type: application/json" -d @apps/app_colette/create_app.json
```

Launch the indexing process:

```bash
curl -X PUT http://localhost:1873/v1/index/app_colette -H "Content-Type: application/json" -d @apps/app_colette/index_app.json
```

Get the status of the indexing process:

```bash
curl -X GET http://localhost:1873/v1/index/app_colette/status
```

Once the indexing process is complete, you can ask questions with REST API:

```bash
curl -X POST  http://localhost:1873/v1/predict/app_colette -H "Content-Type: application/json" -d @apps/app_colette/query_app.json
```

> ⚠️ **Note**
> The response may be very verbose, as it includes the answer along with embedded base64 images used to generate it.



## Adding New Documents to an Existing Index

1. **Place** your new documents in the existing data directory defined in your `index_app.json` file. For example:

```bash
cp apps/app_colette/faq.pdf docs/pdf
```


2. **Re-run** the indexing process with `update_index` enabled:

```bash
curl -X PUT \
  http://localhost:1873/v1/index/app_colette \
  -H "Content-Type: application/json" \
  -d @apps/app_colette/reindex_app.json
```

Inside `reindex_app.json`, ensure you have:

```jsonc
{
  "reindex": false,         // keep existing index data
  "update_index": true,     // add new docs only
  "index_protection": false // allow overwriting mappings/settings
}
```

Only the new files will be indexed.
