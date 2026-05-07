# ── Stage 1: Dependencies ────────────────────────────────────
FROM node:20-alpine AS deps

WORKDIR /app
COPY package.json package-lock.json* ./
RUN npm ci

# ── Stage 2: Build ───────────────────────────────────────────
FROM node:20-alpine AS builder

WORKDIR /app
COPY --from=deps /app/node_modules ./node_modules
COPY . .

# Enable standalone output for Docker (adds output: 'standalone' to next.config)
RUN if ! grep -q "output:" next.config.ts 2>/dev/null && ! grep -q "output:" next.config.js 2>/dev/null; then \
      sed -i 's/const nextConfig: NextConfig = {/const nextConfig: NextConfig = {\n  output: "standalone",/' next.config.ts 2>/dev/null || \
      sed -i 's/const nextConfig = {/const nextConfig = {\n  output: "standalone",/' next.config.js 2>/dev/null; \
    fi

# Build args become env vars at build time for Next.js to inline
ARG NEXT_PUBLIC_AGENT_URL=http://localhost:8000
ARG NEXT_PUBLIC_AGENT_WS_URL=ws://localhost:8000
ARG NEXT_PUBLIC_AUTH_DISABLED=true
ARG NEXT_PUBLIC_DEV_USER_EMAIL=local@simd.local

ENV NEXT_PUBLIC_AGENT_URL=$NEXT_PUBLIC_AGENT_URL \
    NEXT_PUBLIC_AGENT_WS_URL=$NEXT_PUBLIC_AGENT_WS_URL \
    NEXT_PUBLIC_AUTH_DISABLED=$NEXT_PUBLIC_AUTH_DISABLED \
    NEXT_PUBLIC_DEV_USER_EMAIL=$NEXT_PUBLIC_DEV_USER_EMAIL

RUN npm run build

# ── Stage 3: Runtime ─────────────────────────────────────────
FROM node:20-alpine

WORKDIR /app

# Copy standalone build output
COPY --from=builder /app/.next/standalone ./
COPY --from=builder /app/.next/static ./.next/static
COPY --from=builder /app/public ./public

# Runtime env vars (server-side only — not inlined at build time)
ENV NODE_ENV=production \
    PORT=3000 \
    HOSTNAME=0.0.0.0

EXPOSE 3000

CMD ["node", "server.js"]
