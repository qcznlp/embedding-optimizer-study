# Archived PDF and ignored-output delivery correction

Final anonymous-link review found that the reviewed PDF existed locally but was
excluded from the first public payload by the generic build-directory ignore rule.
The scientific/source payload and complete numerical closure were present, but
the README PDF link was not yet usable. This correction does not rewrite that history.

The top-level local archive inventories identify 84 omitted files / 51,747,992
bytes: nine PDFs, their original compilation outputs and eight wheel/sdist pairs.
Every added file matches its pre-existing manifest SHA-256 and byte count. No
source, test, protocol, scientific result, manuscript text or old manifest changes.
The reviewed PDF retains its original SHA-256
7f72c08717e4538cc498259043457f424d598abb92846b261461611b24f46d09.

All added payloads and readable archive members are credential-scanned. Any HF-like
RECORD substring is accepted only after independently recomputing the referenced
member's complete SHA-256 and size. No other match is admitted or pattern relaxed.

The 280 locally unresolved entries all belong to one explicitly external HF artifact
manifest, documented in the second-stage pool-B backup guide at immutable revision
35ce505d3cf3389f4f1a2f35b46259749bc1d7ed. They are not local archive omissions and
are not copied into another namespace. No completed HF download is repeated.

This is a publication correction only. Earlier 3,800 source-version test passes,
complete numerical/paper builds and distribution checks concern unchanged code.
The original failed link check and exact omitted-member inventory are retained here.
