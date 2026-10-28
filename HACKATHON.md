# Faster brain registration with FireANTs and pyAFQ

Every brain has a different shape. Before researchers can compare brain structures, they need to align corresponding anatomy—a process called registration.

FireANTs offers promising GPU acceleration for registration. Our hackathon project asks whether it can provide useful registrations and consistent downstream results inside pyAFQ, a tool for studying the brain’s white-matter pathways.

## Start here

Read [SPEC.md](SPEC.md) for the source-reviewed implementation plan.

Our first milestone is a real-data feasibility experiment: compare registration quality, runtime, and inverse-transform accuracy on one Stanford HARDI subject before building the integration.

**Current status:** specification prepared; real-data feasibility remains untested.

## How to contribute

We welcome help with:

- The initial CPU and GPU experiments.
- Coordinate-conversion and transformation tests.
- Minimal pyAFQ integration.
- Anatomical overlays and downstream validation.

Check this repository’s Issues before starting work. Comment on a task to coordinate with other contributors.
