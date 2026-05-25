package com.genailab.mcp;

import java.util.List;

public record ClaimNormalizationResult(
        String normalizedClaim,
        List<String> keyTerms,
        String confidenceNote) {
}

