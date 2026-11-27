# gAItkeep

**Brand Identification from Product Imagery · Independent Computer Vision Project**

## Overview
gAItkeep is a computer vision system that identifies brand identity from a single product photo, rather than matching an exact SKU. Two products can share the same color, fabric, and even hardware and still be unmistakably different brands once you know what to look for, the small, deliberate design signatures ("brand codes") a house builds its identity around. gAItkeep is trained to recognize those signatures the way a trained eye does.

What started as an extension of computer vision coursework grew into an independent project built well past anything a class required.

## What It Does
Given a photo, gAItkeep classifies which of two brand catalogs (built on real product imagery from Frankies Bikinis and Mirchi by Kim) the item belongs to, learning each brand's visual signature well enough to separate near-identical pieces.

## Architecture
- Every catalog image is encoded with a pretrained MobileNetV2 CNN embedding (TensorFlow), using a strong, efficient visual representation with no training from scratch
- A query photo is compared against the full catalog via cosine similarity, with calibrated thresholds distinguishing "same photo" from "same item" matches
- A top-K brand vote across nearest neighbors, rather than a single best match, keeps an imbalanced catalog (one brand with far more indexed images) from biasing the result
- An offline pipeline (scrape, download, build a persisted embedding index and manifest) feeds a fast online query path, with a batch-testing harness logging every run to CSV
- Shipped end-to-end as a FastAPI/Uvicorn service behind a React + TypeScript interface, returning structured predictions in real time as a deployed, queryable application

## Performance
Like most embedding-based systems, there's a one-time cost: the first request after a cold start is slower while the model loads into memory. Every query after that returns in a fraction of a second. Validated live against photos the model had never seen, correctly separating near-identical pieces by brand in real time.

## Why It Matters
The primary use case is everyday: giving people a fast, reliable way to identify a brand from a single photo, even when it looks nearly identical to something else on the rack. Brands and houses can use the same approach as a check on brand integrity as collections evolve season over season, a meaningful secondary benefit, though adoption at scale will come from solving that everyday moment.

## Next Steps
Expanding the brand catalog deliberately, training and validating accuracy on each addition before any expansion goes live. Longer term, extending to archival and pre-internet pieces, working toward a fully digitized record of fashion history.

## Tech Stack
Python, TensorFlow, NumPy, Pillow, FastAPI/Uvicorn, React + TypeScript (Vite)
