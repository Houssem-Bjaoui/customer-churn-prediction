# Makefile
# ═════════
# Shortcuts for common Docker commands.
# Instead of typing long commands, you type:
#   make build
#   make up
#   make down
#   make logs

# Build all Docker images
build:
	docker compose build

# Start all containers in the background
# -d = detached mode (runs in background)
up:
	docker compose up -d

# Start containers and show logs in terminal
# Good for debugging
up-logs:
	docker compose up

# Stop all containers
down:
	docker compose down

# Stop and remove volumes (clean slate)
clean:
	docker compose down -v --remove-orphans

# Show logs from all containers
logs:
	docker compose logs -f

# Show logs from API only
logs-api:
	docker compose logs -f api

# Show logs from dashboard only
logs-dashboard:
	docker compose logs -f dashboard

# Show running containers
ps:
	docker compose ps

# Rebuild and restart (use after code changes)
rebuild:
	docker compose down
	docker compose build --no-cache
	docker compose up -d

# Run API tests inside the container
test:
	docker compose exec api pytest api/tests/ -v

# Open a shell inside the API container
# Useful for debugging
shell-api:
	docker compose exec api /bin/bash

shell-dashboard:
	docker compose exec dashboard /bin/bash