.PHONY : lint
lint :
	flake8 .

.PHONY : format
format :
	black --check .

.PHONY : typecheck
typecheck :
	mypy . \
		--ignore-missing-imports \
		--no-strict-optional \
		--no-site-packages \
		--cache-dir=/dev/null

.PHONY : test
test :
	pytest --color=yes -rf --durations=40
