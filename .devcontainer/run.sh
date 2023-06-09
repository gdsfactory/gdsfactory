
mkdir -p ~/.config
docker run -it --name code-server -p 127.0.0.1:8080:8080 \
	  -v "$HOME:/home/coder" \
	  -u "$(id -u):$(id -g)" \
	  -e "DOCKER_USER=$USER" \
	  codercom/code-server:latest
