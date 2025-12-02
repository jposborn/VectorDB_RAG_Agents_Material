FROM ollama/ollama:latest

# Default model (can be overridden at runtime)
ENV OLLAMA_MODEL=phi3:mini

# Copy our entrypoint script into the image
COPY entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

EXPOSE 11434

ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
