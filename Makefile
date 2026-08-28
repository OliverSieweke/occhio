.PHONY: docs

docs:
	$(MAKE) -C docs $(filter-out docs,$(MAKECMDGOALS))

%:
	@:
