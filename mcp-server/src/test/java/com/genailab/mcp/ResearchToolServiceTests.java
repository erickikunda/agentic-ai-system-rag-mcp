package com.genailab.mcp;

import static org.assertj.core.api.Assertions.assertThat;

import org.junit.jupiter.api.Test;

class ResearchToolServiceTests {

    private final ResearchToolService service = new ResearchToolService();

    @Test
    void normalizesClaimAndExtractsTerms() {
        ClaimNormalizationResult result = service.normalizeClaim("  embedding drift degrades retrieval quality  ");

        assertThat(result.normalizedClaim()).isEqualTo("embedding drift degrades retrieval quality.");
        assertThat(result.keyTerms()).contains("embedding", "retrieval", "quality");
    }

    @Test
    void classifiesArxivVenue() {
        VenueMetadata metadata = service.lookupVenueMetadata("arXiv");

        assertThat(metadata.venueType()).isEqualTo("preprint repository");
        assertThat(metadata.caution()).contains("corroborating");
    }

    @Test
    void buildsReadingPlan() {
        ReadingPlan plan = service.buildReadingPlan("MCP integration", "experienced Python engineer");

        assertThat(plan.steps()).hasSize(4);
        assertThat(plan.expectedOutcome()).contains("experienced Python engineer");
    }
}

