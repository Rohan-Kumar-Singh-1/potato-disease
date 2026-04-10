# Step 1: Build React app
FROM node:20 AS build

WORKDIR /app

COPY package*.json ./
RUN npm install

COPY . .

# 🔥 IMPORTANT: Inject backend API URL at build time
ENV VITE_API_URL=https://potato-api-gw1e.onrender.com

RUN npm run build

# Step 2: Serve using nginx
FROM nginx:alpine

COPY --from=build /app/dist /usr/share/nginx/html

EXPOSE 80

CMD ["nginx", "-g", "daemon off;"]