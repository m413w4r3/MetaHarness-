MODELS_COMPOSE = docker compose --env-file .env.models -f compose.models.yaml

.PHONY: models-config models-volume models-build models-rebuild models-up models-down models-status models-logs

models-config:
	$(MODELS_COMPOSE) config

models-volume:
	docker volume inspect chatgpt-bridge_bridge_data >/dev/null 2>&1 \
	  || docker volume create chatgpt-bridge_bridge_data

models-build:
	$(MODELS_COMPOSE) build

models-rebuild: models-volume
	$(MODELS_COMPOSE) build
	$(MODELS_COMPOSE) up -d --no-build --pull never

models-up: models-volume
	$(MODELS_COMPOSE) up -d --no-build --pull never

models-down:
	$(MODELS_COMPOSE) down

models-status:
	$(MODELS_COMPOSE) ps

models-logs:
	$(MODELS_COMPOSE) logs -f --tail=100
