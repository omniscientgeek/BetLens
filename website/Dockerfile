# BetStamp Website — React build + nginx static serving
# ----------------------------------------------------------
# Build context: project root (/)
# Build args (set in Railway dashboard):
#   REACT_APP_API_BASE   — e.g. https://betstamp-api.up.railway.app/api
#   REACT_APP_SOCKET_URL — e.g. https://betstamp-api.up.railway.app
# ----------------------------------------------------------

FROM node:20-alpine AS build

WORKDIR /app

# Install dependencies first (cache layer)
COPY website/package.json website/package-lock.json ./
RUN npm ci

# Copy source
COPY website/ .

# Inject API URLs at build time (Create React App reads REACT_APP_* from env)
ARG REACT_APP_API_BASE
ARG REACT_APP_SOCKET_URL
ENV REACT_APP_API_BASE=$REACT_APP_API_BASE
ENV REACT_APP_SOCKET_URL=$REACT_APP_SOCKET_URL

RUN npm run build

# ---- Serve with nginx ----
FROM nginx:stable-alpine

# Install envsubst (part of gettext)
RUN apk add --no-cache gettext

COPY --from=build /app/build /usr/share/nginx/html

# Use a template for nginx config (envsubst replaces $PORT)
COPY website/nginx.conf /etc/nginx/templates/default.conf.template

EXPOSE 80

# nginx docker image auto-processes /etc/nginx/templates/*.template with envsubst
# but we need to control which vars get substituted (avoid replacing $uri etc.)
CMD ["sh", "-c", "export PORT=${PORT:-80} && envsubst '${PORT}' < /etc/nginx/templates/default.conf.template > /etc/nginx/conf.d/default.conf && nginx -g 'daemon off;'"]
