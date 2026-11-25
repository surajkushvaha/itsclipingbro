"""HTTP layer of clipbro. Requires the optional ``[api]`` extra (FastAPI).

Only ``clipbro.api.motion_routes`` exists today. It exposes the motion analyzer over HTTP and
is served with ``uvicorn clipbro.api.motion_routes:create_app --factory``.
"""
