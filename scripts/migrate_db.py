"""在项目目录的 PowerShell 显式迁移。首次旧库接管必须先备份，再传 --adopt-existing。"""

import argparse
import asyncio

from python_ai.db.migrations import upgrade
from python_ai.db.session import dispose_db, engine


async def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adopt-existing", action="store_true")
    args = parser.parse_args()
    try:
        async with engine().begin() as connection:
            await connection.run_sync(
                lambda conn: upgrade(conn, adopt_existing=args.adopt_existing)
            )
        print("数据库业务表已迁移到最新版本。")
    finally:
        await dispose_db()


if __name__ == "__main__":
    asyncio.run(main())
