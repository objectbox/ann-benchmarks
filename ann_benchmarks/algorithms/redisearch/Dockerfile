FROM redis/redis-stack-server as redis-server
FROM ann-benchmarks

# Install Redis
RUN apt-get update && apt-get install -y curl lsb-core
RUN curl -fsSL https://packages.redis.io/gpg | gpg --dearmor -o /usr/share/keyrings/redis-archive-keyring.gpg
RUN echo "deb [signed-by=/usr/share/keyrings/redis-archive-keyring.gpg] https://packages.redis.io/deb $(lsb_release -cs) main" | tee /etc/apt/sources.list.d/redis.list
RUN apt-get update
RUN apt-get install -y redis

# Copy redisearch module from /redis/redis-stack-server build image
COPY --from=redis-server /opt/redis-stack/lib/redisearch.so /opt/redis-stack/lib/redisearch.so

RUN pip3 install redis
