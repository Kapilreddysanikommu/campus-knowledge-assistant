import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// The dev server proxies /query to the FastAPI backend, so the browser
// only ever talks to one origin and no CORS setup is needed on the
// backend.
export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      "/query": "http://127.0.0.1:8000",
    },
  },
});
