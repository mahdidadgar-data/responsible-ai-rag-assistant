# Data Folder

This folder is used for local project data during development.

## Folder Structure

```text
data/
├── raw/
├── processed/
├── vector_store/
└── README.md
```

## Folder Purpose

### `data/raw/`

This folder is for locally stored raw source files, such as downloaded PDFs or extracted web text.

Raw source files should not be uploaded to GitHub.

### `data/processed/`

This folder is for cleaned and chunked text files created during the document-processing pipeline.

Processed files should be checked carefully before deciding whether they are safe and appropriate to upload.

### `data/vector_store/`

This folder is for the local Chroma vector database.

Vector-store files should not be uploaded to GitHub because they can be large and are generated artifacts.

## GitHub Policy

The project should not upload:

* raw downloaded documents
* private documents
* API keys
* `.env` files
* generated vector stores
* confidential data
* personal data

The project may upload:

* this `README.md`
* source metadata
* project documentation
* evaluation question templates
* code used to recreate the pipeline

## Reproducibility

The source documents used in this project are listed in:

```text
docs/source_register.md
```

The processing pipeline should allow another user to recreate the processed dataset locally from the public source list.
