# flowgate — Product Requirements (v1)

## Purpose

flowgate is a reliability gateway that sits in front of a backend handler and protects it with load shedding, per-client rate limiting and a circuit breaker.

## Endpoints

- GET /healthz returns 200 with body "ok".
- GET /work passes through the protection chain to a simulated backend.
- GET /metrics exposes Prometheus text-format metrics.

## Protection chain

Requests to /work pass through the checks in this order: load shedding, then rate limiting, then the circuit breaker, then the backend.

### Load shedding

When 50 requests are already in flight, additional requests are rejected immediately with 503 and body "server overloaded, try again shortly". Requests are never queued.

### Rate limiting

Each client is limited to 5 requests per second sustained, with a burst of 20.

A client is identified by the Fly-Client-IP header when present, otherwise by the IP address of the connection, ignoring the port.

Over-limit requests are rejected with 429, body "rate limit exceeded", and a Retry-After header of at least 1 second.

### Circuit breaker

After 5 backend failures the breaker opens. While open, requests are rejected with 503 and body "backend unavailable". After a 10 second cooldown the breaker allows a trial request; success closes it, failure reopens it.

### Simulated backend

The backend responds in 5 to 20 milliseconds. About 2% of requests fail with 500.

## Observability

/metrics reports flowgate_requests_total with an outcome label (allowed, error, shed, rate_limited, breaker_open), flowgate_breaker_state, and flowgate_in_flight_requests.

## Service level objectives

- p95 latency of /work at or below 150 ms under about 2,000 requests per second sustained.
- Backend success rate of at least 95% for admitted requests.

## Known limitations

Rate-limit, breaker and metrics state is held in memory per instance. With more than one instance, a client's effective rate limit is multiplied by the number of instances it reaches.
