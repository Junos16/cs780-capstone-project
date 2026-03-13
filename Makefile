# Default values for arguments
AGENT ?= ddqn
SUBMISSION ?= 1.ddqn_provided_sample
LEVEL ?= 1
EPISODES ?= 100
WALL ?= ""

# Check if WALL is provided to append the corresponding flag
WALL_FLAG = $(if $(filter-out "",$(WALL)),--wall,)

.PHONY: train eval help

## Train an agent
train:
	uv run src/main.py train --agent $(AGENT) --level $(LEVEL) --episodes $(EPISODES) $(WALL_FLAG)

## Evaluate a submission
eval:
	uv run src/main.py eval --submission $(SUBMISSION) --level $(LEVEL) --episodes $(EPISODES) $(WALL_FLAG)

help:
	@echo "Usage:"
	@echo "  make train [AGENT=name] [LEVEL=1|2|3] [EPISODES=N] [WALL=1]"
	@echo "    Example: make train AGENT=ddqn LEVEL=2 EPISODES=1000"
	@echo ""
	@echo "  make eval [SUBMISSION=name] [LEVEL=1|2|3] [EPISODES=N] [WALL=1]"
	@echo "    Example: make eval SUBMISSION=1.ddqn_provided_sample LEVEL=1 EPISODES=10"
