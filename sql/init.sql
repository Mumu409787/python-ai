-- 首次启动容器时由 docker-entrypoint-initdb.d 自动执行，只跑一次。
-- 注意：initdb 脚本只在数据卷为空时执行；改了这个文件后要重跑，
--       必须先 `docker compose down -v` 清空数据卷，否则不会生效。

-- 启用 pgvector 扩展（镜像 pgvector/pgvector 已内置，这里只是启用）
CREATE EXTENSION IF NOT EXISTS vector;

-- 自检：应返回 vector | <版本号>
-- SELECT extname, extversion FROM pg_extension WHERE extname = 'vector';
