.PHONY: docs

docs:
	make -C docs $(filter-out docs,$(MAKECMDGOALS))

%:
	@: