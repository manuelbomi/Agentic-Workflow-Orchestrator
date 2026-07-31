"""Concrete agent implementations for the research-to-memo pipeline.

Pipeline order: ``researcher_agent`` -> ``analyst_agent`` -> ``reviewer_agent``
(human-in-the-loop gate) -> ``writer_agent``.
"""
