install:
	pip install hupper

debug2:
	hupper -m waitress --port=8000 myapp:application

debug:
	bash debug.sh

production:
	export FLASK_ENV=production
	flask run
