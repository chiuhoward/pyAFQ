# Faster brain registration with FireANTs and pyAFQ

Every brain has a different shape. Before researchers can compare brain structures, they need to align corresponding anatomy—a process called registration.

FireANTs offers GPU acceleration for registration. Our hackathon project integrates it into pyAFQ, a tool for studying the brain’s white-matter pathways, with minimal additional code, correct image and streamline transforms, and measured runtime benefit.

## Start here

Read [SPEC.md](SPEC.md) for the source-reviewed implementation plan.

Build the smallest optional Python integration using existing FireANTs, DIPY and pyAFQ functionality. Add focused tests for the new boundaries and reuse existing coverage. Validate correctness and complete registration runtime against SynMap on one Stanford HARDI subject, using the same harness throughout development.

**Current status:** coordinate decoder and its analytic tests merged in PR #6. The mapping class, end-to-end correctness, and performance measurements remain to be implemented. Retain the decoder only if the selected integration uses it.

## How to contribute

We welcome help with:

- The initial CPU and GPU experiments.
- Coordinate-conversion and transformation tests.
- Minimal pyAFQ integration.
- Anatomical overlays and downstream validation.

Check this repository’s Issues before starting work. Comment on a task to coordinate with other contributors.
