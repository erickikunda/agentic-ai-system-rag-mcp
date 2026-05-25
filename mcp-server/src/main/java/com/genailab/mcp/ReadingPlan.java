package com.genailab.mcp;

import java.util.List;

public record ReadingPlan(
        String topic,
        List<String> steps,
        String expectedOutcome) {
}

