# VideoSlack — frontend

Next.js 16 app for VideoSlack. See the [main README](../README.md) for setup.

```bash
npm install
npm run dev      # http://localhost:3000
```

Set `NEXT_PUBLIC_API_URL` in `.env.local` if the backend isn't on `http://localhost:8000`.

- `app/page.tsx` — page state (loading / form / dashboard)
- `components/` — `IngestForm`, `VideoCard`, `ComparisonPanel`, `ChatPanel`, `Markdown` (renders `[A 0:42]` citations as links)
- `lib/api.ts` — API client, including the `fetch`-based SSE stream parser
