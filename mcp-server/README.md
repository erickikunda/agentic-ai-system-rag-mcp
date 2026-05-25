# genai-lab MCP server

Spring Boot MCP server for the research assistant domain tools.

## Tools

- `normalize_claim`
- `lookup_venue_metadata`
- `build_reading_plan`

## Run

```bash
mvn test
mvn spring-boot:run
```

The server uses Spring AI tool annotations and the WebFlux MCP server starter,
which auto-converts registered Spring AI tools into MCP tool specifications.
