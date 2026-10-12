# flowgate — test environment contract

What a test runner may assume about the system under test. This is not product truth: requirement quotes must come from the PRD, never from this file.

## Environments

| Name | Target | Notes |
|---|---|---|
| live | https://flowgate-pramathesh.fly.dev | Deployed. Configuration is fixed; env overrides are not available. Fly's edge sets Fly-Client-IP, so a test cannot choose its own client identity. |
| local | http://localhost:8080 | Started per test with chosen configuration. Fly-Client-IP is honored as supplied, so a test can choose client identities. |

## Starting a local instance

Branch `resilience-experiments` of github.com/prime5/flowgate:

```bash
MAX_IN_FLIGHT=2 BREAKER_THRESHOLD=1 go run ./cmd/flowgate
```

## Configuration knobs (local only)

| Variable | Default | Effect |
|---|---|---|
| PORT | 8080 | Listen port |
| RL_BURST | 20 | Token bucket capacity per client |
| RL_RATE | 5 | Sustained tokens per second per client |
| BREAKER_THRESHOLD | 5 | Backend failures before the breaker opens |
| MAX_IN_FLIGHT | 50 | In-flight requests allowed before shedding |

The breaker cooldown (10s) and the simulated backend (5-20ms latency, 2% error rate) are compiled in and cannot be configured.

## Consequences for test design

- Lowering `MAX_IN_FLIGHT` makes load shedding reachable without generating hundreds of concurrent clients.
- Setting `BREAKER_THRESHOLD=1` makes the breaker open on the first backend failure, which the 2% simulated error rate will produce within a few hundred requests. This is the only way to drive the breaker black-box.
- Raising `RL_BURST` prevents the rate limiter from rejecting traffic intended to exercise shedding or the breaker.
- One instance only. Rate-limit, breaker and metrics state is per instance.
