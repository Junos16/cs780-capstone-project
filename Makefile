# Default values for arguments
AGENT ?= ddqn
SUBMISSION ?= 1.ddqn_provided_sample
LEVEL ?= 1
EPISODES ?= 100
WALL ?= ""
CONFIG ?= ""
TRIALS ?= 30

# Check if flags are provided to append them
WALL_FLAG = $(if $(filter-out "",$(WALL)),--wall,)
CONFIG_FLAG = $(if $(filter-out "",$(CONFIG)),--config $(CONFIG),)

.PHONY: sweep train eval submit help

## Sweep an agent
sweep:
	uv run src/main.py sweep --agent $(AGENT) --level $(LEVEL) --episodes $(EPISODES) --trials $(TRIALS) $(WALL_FLAG)

## Train an agent
train:
	uv run src/main.py train --agent $(AGENT) --level $(LEVEL) --episodes $(EPISODES) $(WALL_FLAG) $(CONFIG_FLAG)

## Evaluate a submission
eval:
	uv run src/main.py eval --submission $(SUBMISSION) --level $(LEVEL) --episodes $(EPISODES) $(WALL_FLAG)

## Package a submission into a zip file
submit:
	@if [ ! -d "submissions/$(SUBMISSION)" ]; then echo "Error: submissions/$(SUBMISSION) does not exist"; exit 1; fi
	@cd submissions/$(SUBMISSION) && zip submission.zip agent.py weights.pth
	@echo "Created submissions/$(SUBMISSION)/submission.zip"

help:
	@echo "Usage:"
	@echo "  make sweep [AGENT=name] [LEVEL=1|2|3] [EPISODES=N] [WALL=1] [TRIALS=N]"
	@echo "    Example: make sweep AGENT=sarsa_lambda LEVEL=1 EPISODES=300 TRIALS=30 WALL=1"
	@echo ""
	@echo "  make train [AGENT=name] [LEVEL=1|2|3] [EPISODES=N] [WALL=1] [CONFIG=path/to/json]"
	@echo "    Example: make train AGENT=ddqn LEVEL=2 EPISODES=1000 CONFIG=submissions/configs/1.ddqn_provided_sample.json"
	@echo ""
	@echo "  make eval [SUBMISSION=name] [LEVEL=1|2|3] [EPISODES=N] [WALL=1]"
	@echo "    Example: make eval SUBMISSION=1.ddqn_provided_sample LEVEL=1 EPISODES=10"
	@echo ""
	@echo "  make submit [SUBMISSION=name]"
	@echo "    Example: make submit SUBMISSION=1.ddqn_provided_sample"
