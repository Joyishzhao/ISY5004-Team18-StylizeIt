"""StylizeIt processing pipeline.

The pipeline is split into small, replaceable stages (ingest -> grounding ->
tracking -> generation -> temporal -> compositor -> export -> evaluation),
each living in its own file. The orchestrator wires them together.
"""
