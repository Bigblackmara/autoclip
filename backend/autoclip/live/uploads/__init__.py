"""Upload subsystem for Live Watch (A7 owns bodies).

youtube.py    — YouTube Data API v3 uploads (official API, OAuth installed-app flow)
instagram.py  — Instagram Graph API (professional account, container upload)
tokens.py     — OAuth token storage in the OS keyring (access+refresh blobs)

Hard rules (from the task spec): official APIs only, no browser automation,
tokens/credentials in the OS keyring, never in config files or logs.
"""
