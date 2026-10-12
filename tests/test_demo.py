"""演示页面、样例与指南都由同一 FastAPI 应用提供。"""

import httpx

from python_ai.main import app


async def test_demo_pages_and_assets():
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        for path, expected in [
            ("/", "让资料，成为答案。"),
            ("/compare", "先用同一个问题"),
            ("/demo/sample", "Token 有效期为 2 小时"),
            ("/demo/guide", "案例 05：多轮记忆"),
            ("/demo/guide?stages=true", "阶段①"),
            ("/demo/operations", "运行手册"),
            ("/demo/operations?document=troubleshoot", "故障排查"),
            ("/demo/operations?document=verify", "案例 05：多轮记忆"),
            ("/demo/operations?document=stages", "阶段①"),
            ("/demo/operations?document=compare", "本机双版本对照"),
            ("/demo/operations?document=records", "验证记录与限制"),
            ("/demo/comparison-guide", "本机双版本对照"),
            ("/static/demo/comparison-guide.md", "本机双版本对照"),
            ("/static/demo/app.js", "parseSSE"),
            ("/static/demo/style.css", "workspace-grid"),
        ]:
            response = await client.get(path)
            assert response.status_code == 200
            assert expected in response.text
        for kind, header in [("mixed", b"%PDF"), ("scan", b"%PDF"), ("image", b"\x89PNG")]:
            response = await client.get(f"/demo/ocr-sample?kind={kind}")
            assert response.status_code == 200
            assert response.content.startswith(header)
        invalid = await client.get("/demo/ocr-sample?kind=../../.env")
        assert invalid.json()["code"] == 1001
