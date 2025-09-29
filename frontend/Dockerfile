# ---- build stage ----
FROM node:18-alpine AS build
WORKDIR /app

COPY package*.json ./
RUN npm ci

# CRA берёт только REACT_APP_* ; вызовы пойдут на /api
ARG REACT_APP_API_BASE_URL="/api"
ENV REACT_APP_API_BASE_URL=${REACT_APP_API_BASE_URL}

COPY . .
RUN npm run build

# ---- production stage ----
FROM nginx:alpine

COPY --from=build /app/build /usr/share/nginx/html

# Шаблон nginx, который на старте превратится в /etc/nginx/conf.d/default.conf
RUN mkdir -p /etc/nginx/templates && cat > /etc/nginx/templates/default.conf.template <<'NGINX'
server {
  listen 80;
  server_name _;
  root /usr/share/nginx/html;
  index index.html;

  # Разрешаем крупные файлы (подгони под свой лимит)
  client_max_body_size 200m;

  # Docker DNS + динамический резолв
  resolver 127.0.0.11 ipv6=off valid=10s;
  set $api_upstream "${API_HOST}:${API_PORT}";

  location ${API_PREFIX}/ {
    # ВАЖНО: без завершающего / — чтобы не срезать /api в URI
    proxy_pass http://${api_upstream};

    # Для больших аплоадов/долгой обработки
    proxy_request_buffering off;   # стримим тело сразу в бэкенд
    proxy_buffering off;           # если бэкенд шлёт стрим/прогресс
    proxy_read_timeout  600s;
    proxy_send_timeout  600s;
    proxy_connect_timeout 60s;

    proxy_set_header Host $host;
    proxy_set_header X-Real-IP $remote_addr;
    proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
    proxy_set_header X-Forwarded-Proto $scheme;
  }

  location / { try_files $uri /index.html; }

  location ~* \.(?:js|css|ico|png|jpg|jpeg|gif|svg|woff2?)$ {
    try_files $uri =404;
    access_log off;
    add_header Cache-Control "public, max-age=31536000, immutable";
  }
}

NGINX

ENV NGINX_ENVSUBST_TEMPLATE_DIR=/etc/nginx/templates
ENV NGINX_ENVSUBST_TEMPLATE_SUFFIX=.template

EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]
