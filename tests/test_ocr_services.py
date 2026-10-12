"""真实本地 OCR；服务联调只写本次临时文档/集合，无 API 费用。"""

import os
from uuid import uuid4

import pytest

from python_ai import ingest, lexical, parsing, retrieval, vectorstore
from python_ai.clients import close_clients, qdrant
from python_ai.config import PROJECT_ROOT, get_settings
from python_ai.db.session import dispose_db, init_db

pytestmark = pytest.mark.skipif(os.getenv("RUN_OCR_TESTS") != "1", reason="需开启真实 OCR 验收")


@pytest.mark.parametrize("name", ["crane-scan.png", "crane-scanned.pdf", "crane-mixed.pdf"])
def test_real_ocr_pages(name):
    path = PROJECT_ROOT / "data/samples/ocr" / name
    pages = parsing.parse_pages(name, path.read_bytes())
    assert "CRANE-18743" in pages[-1].text and "9" in pages[-1].text
    assert pages[-1].source_page == (2 if name == "crane-mixed.pdf" else 1)
    if name == "crane-mixed.pdf":
        assert "29012" in pages[0].text and pages[0].source_page == 1


@pytest.mark.skipif(os.getenv("RUN_SERVICE_TESTS") != "1", reason="还需开启 PG/Qdrant 服务验收")
async def test_ocr_ingest_retrieve_page_and_delete(monkeypatch):
    token = uuid4().hex
    collection = "test_ocr_" + token
    monkeypatch.setattr(get_settings(), "qdrant_collection", collection)
    monkeypatch.setattr(get_settings(), "rerank_enabled", False)

    async def embed(texts):
        return [[1.0, 0.0, 0.0, 0.0] for _ in texts]

    async def query(text):
        return [1.0, 0.0, 0.0, 0.0]

    async def dimension():
        return 4

    monkeypatch.setattr(ingest, "embed_texts", embed)
    monkeypatch.setattr(vectorstore, "embed_query", query)
    monkeypatch.setattr(vectorstore, "probe_dimension", dimension)
    doc = None
    try:
        await init_db()
        await vectorstore.ensure_collection()
        path = PROJECT_ROOT / "data/samples/ocr/crane-mixed.pdf"
        data = path.read_bytes() + b"\n% " + token.encode()
        result = await ingest.ingest_document("qa-" + token + ".pdf", data)
        assert not result.duplicated
        doc = result.doc_id
        hits = await retrieval.retrieve("CRANE-18743")
        selected = [h for h in hits if h.doc_id == doc and "CRANE-18743" in h.content]
        assert selected and selected[0].source_page == 2
        assert retrieval.to_dtos(selected)[0].source_page == 2
        await ingest.delete_document(doc)
        assert await vectorstore.count_by_doc(doc) == 0
        assert all(h.doc_id != doc for h in await lexical.bm25_recall("CRANE-18743", top_k=20))
        doc = None
    finally:
        try:
            if doc is not None:
                await ingest.delete_document(doc)
            if await qdrant().collection_exists(collection):
                await qdrant().delete_collection(collection)
        finally:
            await dispose_db()
            await close_clients()
