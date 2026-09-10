MODELS_COMPOSE = docker compose --env-file .env.models -f compose.models.yaml

.PHONY: models-config models-build models-rebuild models-up models-down models-status models-logs

models-config:
	$(MODELS_COMPOSE) config

models-build:
	$(MODELS_COMPOSE) build

models-rebuild:
	$(MODELS_COMPOSE) build
	$(MODELS_COMPOSE) up -d

models-up:
	$(MODELS_COMPOSE) up -d --no-build --pull never

models-down:
	$(MODELS_COMPOSE) down

models-status:
	$(MODELS_COMPOSE) ps

models-logs:
	$(MODELS_COMPOSE) logs -f --tail=100
