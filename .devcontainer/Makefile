
build:
	docker build -t gdsfactory -f Dockerfile.dev .

build-web:
	docker build -t gdsfactory-web -f Dockerfile.web .

run:
	docker run -p 8080:8080 -p 8000:8000 gdsfactory-web

vscode:
	bash run.sh
